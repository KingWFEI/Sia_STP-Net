"""
run_ablation.py  —  自动运行全部消融实验

消融实验流水线:
  1-Baseline     Siamese Encoder-Decoder (无任何增强)
  2-STE          + 浅层时间增强 (Shallow Temporal Enhancement)
  3-STN          + 深层 STN 特征对齐
  4-BDF          + BiGatedDifferenceFusion
  5-PGD          + 时空先验引导解码器 (完整模型)
  6-BiConvLSTM   将 BDF 替换为 Bi-ConvLSTM (与 PGD 同级对比)

用法:
    python run_ablation.py --train_dir DATA/train --val_dir DATA/val \\
                           --base_save_dir ./runs/ablation_study

可选: 只跑部分消融实验
    python run_ablation.py --train_dir DATA/train --val_dir DATA/val \\
                           --ablations baseline stn_mean full

已完成实验自动跳过 (检测 best_dice.pth), 加 --force 可强制重跑。
"""

import argparse
import csv
import gc
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import GradScaler, autocast, nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm.auto import tqdm

# -------------------------------------------------------------------
# 从训练脚本引入通用组件
# -------------------------------------------------------------------
from train import (
    safe_collate,
    count_parameters,
    WarmupCosineScheduler,
    ImprovedEarlyStopping,
    TrainingVisualizer,
    compute_all_metrics,
    train_one_epoch,
    validate_one_epoch,
    log_to_file,
)
from utils.utils import SequencePMTMDataset

# -------------------------------------------------------------------
# 消融网络模型构建
# -------------------------------------------------------------------
from net.sia_prompt_net_ablation import build_model, ABLATION_PRESETS

# -------------------------------------------------------------------
# 消融实验信息定义
# -------------------------------------------------------------------
ABLATION_PRESET_ORDER = [
    "baseline",
    "shallow_temporal",
    "stn_mean",
    "bdf_no_prior",
    "full",
    "biconvlstm",
]

ABLATION_INFO = {
    "baseline":         {"display": "1-Baseline",     "desc": "Siamese Encoder-Decoder (no enhancements)"},
    "shallow_temporal": {"display": "2-STE",          "desc": "+ Shallow Temporal Enhancement"},
    "stn_mean":         {"display": "3-STN",          "desc": "+ STN Feature Alignment"},
    "bdf_no_prior":     {"display": "4-BDF",          "desc": "+ BiGatedDifferenceFusion"},
    "full":             {"display": "5-PGD",          "desc": "Full model + Prior Guided Decoder"},
    "biconvlstm":       {"display": "6-BiConvLSTM",   "desc": "Bi-ConvLSTM variant (replaces BDF)"},
}

# ===================================================================
#  Loss (与训练脚本 BaselineLossWrapper 一致，独立引入避免循环依赖)
# ===================================================================
from loss.loss import LabelSmoothingBCE


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, eps=1e-7):
        pred = pred.sigmoid()
        num = 2 * (pred * target).sum(dim=(2, 3))
        den = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 1 - (num + eps) / (den + eps)
        return dice.mean()


class AblationLossWrapper(nn.Module):
    """与 BaselineLossWrapper 一致的损失组合 (LabelSmoothingBCE + DiceLoss + 深层监督)"""

    def __init__(self, pos_weight, smoothing=0.1):
        super().__init__()
        self.criterion_bce = LabelSmoothingBCE(pos_weight=pos_weight, smoothing=smoothing)
        self.criterion_dice = DiceLoss()

    def _calc_single(self, pred, target):
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
        return 0.5 * self.criterion_bce(pred, target) + 0.5 * self.criterion_dice(pred, target)

    def forward(self, outputs, targets):
        if not isinstance(outputs, dict):
            return self._calc_single(outputs, targets)

        loss = self._calc_single(outputs["seg"], targets)

        if "aux" in outputs:
            aux_weights = [0.4, 0.2, 0.1]
            for i, aux_pred in enumerate(outputs["aux"]):
                w = aux_weights[i] if i < len(aux_weights) else 0.1
                loss += w * self._calc_single(aux_pred, targets)

        return loss


# ===================================================================
#  训练单个消融实验
# ===================================================================
def train_ablation_experiment(args, ablation_name, device):
    info = ABLATION_INFO[ablation_name]
    save_dir = os.path.join(args.base_save_dir, ablation_name)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train_log.txt")

    log_to_file(log_path, f"\n{'=' * 60}")
    log_to_file(log_path, f"Starting Ablation: {info['display']} — {info['desc']}")
    log_to_file(log_path, f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_to_file(log_path, f"{'=' * 60}\n")

    # ---- Data ----
    train_dataset = SequencePMTMDataset(
        root_dir=args.train_dir,
        window_size=args.window_size,
        img_size=args.img_size,
        use_polar=args.use_polar,
        is_train=True,
    )
    val_dataset = SequencePMTMDataset(
        root_dir=args.val_dir,
        window_size=args.window_size,
        img_size=args.img_size,
        use_polar=args.use_polar,
        is_train=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=safe_collate,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=safe_collate,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---- Model ----
    log_to_file(log_path, f"Building model: ablation_name={ablation_name}")
    model = build_model(
        window_size=args.window_size,
        image_size=(args.img_size, args.img_size),
        num_classes=1,
        input_channels=1,
        ablation_name=ablation_name,
        deep_supervision=True,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    log_to_file(
        log_path,
        f"  Total Params: {total_params / 1e6:.2f}M  |  "
        f"Trainable: {trainable_params / 1e6:.2f}M",
    )

    # ---- Optimizer & Loss ----
    use_amp = (device.type == "cuda") and args.amp
    scaler = GradScaler("cuda", enabled=use_amp)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    pos_w = torch.tensor([args.pos_weight], device=device)
    criterion = AblationLossWrapper(pos_weight=pos_w, smoothing=0.1).to(device)

    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=5, total_epochs=args.epochs, min_lr=1e-6
    )
    early_stopper = ImprovedEarlyStopping(patience=args.early_stopping)
    visualizer = TrainingVisualizer(save_dir)

    # ---- Training Loop ----
    best_val_dice = -1.0
    best_val_recall = -1.0
    best_val_iou = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t_loss, t_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            args.epochs,
            scaler,
            use_amp,
            accumulation_steps=args.accumulation_steps,
        )
        v_loss, v_metrics = validate_one_epoch(
            model, val_loader, criterion, device, epoch, args.epochs, use_amp
        )

        t_metrics["loss"] = t_loss
        v_metrics["loss"] = v_loss
        scheduler.step()
        curr_lr = scheduler.get_last_lr()[0]
        visualizer.update(epoch, t_metrics, v_metrics, curr_lr)

        log_msg = (
            f"[{info['display']}] Epoch {epoch}/{args.epochs}  "
            f"(LR: {curr_lr:.6f}):\n"
            f"  Train — Loss: {t_loss:.4f}  Dice: {t_metrics['dice']:.4f}  "
            f"IoU: {t_metrics['iou']:.4f}  Recall: {t_metrics['recall']:.4f}\n"
            f"  Val   — Loss: {v_loss:.4f}  Dice: {v_metrics['dice']:.4f}  "
            f"IoU: {v_metrics['iou']:.4f}  Recall: {v_metrics['recall']:.4f}"
        )
        log_to_file(log_path, log_msg)

        # save last checkpoint
        torch.save(model.state_dict(), os.path.join(save_dir, "last.pth"))

        # best by Dice
        if v_metrics["dice"] > best_val_dice:
            best_val_dice = v_metrics["dice"]
            best_val_iou = v_metrics["iou"]
            best_val_recall = v_metrics["recall"]
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, "best_dice.pth"))
            log_to_file(
                log_path,
                f"  [!] New Best — Dice: {best_val_dice:.4f}  "
                f"IoU: {best_val_iou:.4f}  Recall: {best_val_recall:.4f}  @ epoch {epoch}",
            )

        # best by Recall
        if v_metrics["recall"] > best_val_recall:
            best_val_recall = v_metrics["recall"]
            torch.save(model.state_dict(), os.path.join(save_dir, "best_recall.pth"))

        if early_stopper(v_metrics, epoch, model):
            log_to_file(log_path, f"[Early Stop] Triggered at epoch {epoch}")
            break

        gc.collect()
        torch.cuda.empty_cache()

    visualizer.plot(best_epoch=best_epoch)
    log_to_file(log_path, f"Training Finished. Best Dice: {best_val_dice:.4f} @ epoch {best_epoch}")

    return {
        "ablation": ablation_name,
        "display": info["display"],
        "params_m": total_params / 1e6,
        "best_epoch": best_epoch,
        "best_dice": best_val_dice,
        "best_iou": best_val_iou,
        "best_recall": best_val_recall,
    }


# ===================================================================
#  Main
# ===================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation Study Runner — 自动消融实验",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # data
    parser.add_argument("--train_dir", type=str, required=True,
                        help="训练集根目录")
    parser.add_argument("--val_dir", type=str, required=True,
                        help="验证集根目录")
    parser.add_argument("--base_save_dir", type=str, default="./runs/ablation_study",
                        help="所有消融实验的根保存目录 (每个实验独立子目录)")
    parser.add_argument("--window_size", type=int, default=3,
                        help="时序窗口大小 T")
    parser.add_argument("--img_size", type=int, default=512,
                        help="输入图像尺寸 (正方形)")
    parser.add_argument("--use_polar", action="store_true", default=False,
                        help="是否使用极坐标变换")

    # training
    parser.add_argument("--batch_size", type=int, default=2,
                        help="单卡 batch size")
    parser.add_argument("--accumulation_steps", type=int, default=8,
                        help="梯度累积步数")
    parser.add_argument("--epochs", type=int, default=100,
                        help="最大训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="初始学习率")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pos_weight", type=float, default=2.0)
    parser.add_argument("--early_stopping", type=int, default=30,
                        help="早停耐心值")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="启用混合精度训练")
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    # ablation selection
    parser.add_argument(
        "--ablations", type=str, nargs="+",
        default=ABLATION_PRESET_ORDER,
        choices=list(ABLATION_PRESETS.keys()),
        help="指定要运行的消融实验列表 (默认全部)",
    )

    # resume / force
    parser.add_argument("--force", action="store_true", default=False,
                        help="强制重跑所有实验 (默认检测 best_dice.pth 跳过已完成实验)")

    # gpu
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU 设备编号")

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.base_save_dir, exist_ok=True)

    print("=" * 60)
    print("  ABLATION STUDY — Automatic Runner")
    print(f"  Device       : {device}")
    if torch.cuda.is_available():
        print(f"  GPU          : {torch.cuda.get_device_name(args.gpu)}")
    print(f"  Save Dir     : {args.base_save_dir}")
    print(f"  Experiments  : {len(args.ablations)}")
    for a in args.ablations:
        info = ABLATION_INFO[a]
        print(f"    {info['display']:<16} {info['desc']}")
    print("=" * 60)

    # summary CSV
    summary_path = os.path.join(args.base_save_dir, "ablation_summary.csv")
    summary_fields = [
        "ablation", "display", "params_m", "best_epoch",
        "best_dice", "best_iou", "best_recall",
    ]
    with open(summary_path, "w", newline="") as f:
        csv.writer(f).writerow(summary_fields)

    all_results = []
    start_time = datetime.now()

    for i, ablation_name in enumerate(args.ablations):
        info = ABLATION_INFO[ablation_name]
        save_dir = os.path.join(args.base_save_dir, ablation_name)

        # skip completed experiment
        best_path = os.path.join(save_dir, "best_dice.pth")
        if not args.force and os.path.exists(best_path):
            print(f"\n[Skip] {info['display']} — already completed (use --force to re-run)")
            continue

        print(f"\n{'#' * 60}")
        print(f"#  Experiment {i + 1}/{len(args.ablations)}: {info['display']}")
        print(f"#  {info['desc']}")
        print(f"{'#' * 60}\n")

        result = train_ablation_experiment(args, ablation_name, device)
        all_results.append(result)

        # append to summary CSV
        with open(summary_path, "a", newline="") as f:
            csv.writer(f).writerow([result[k] for k in summary_fields])

        elapsed = datetime.now() - start_time
        print(f"\n>>> Completed: {info['display']}  "
              f"Best Dice: {result['best_dice']:.4f}  "
              f"Best IoU: {result['best_iou']:.4f}  "
              f"(elapsed: {elapsed})")

    # ---- Final Summary ----
    # re-read CSV for all completed results (including pre-skipped ones)
    completed = []
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for k in summary_fields[3:]:
                    row[k] = float(row[k])
                completed.append(row)

    if completed:
        print("\n" + "=" * 75)
        print("  ABLATION STUDY — FINAL SUMMARY")
        print("=" * 75)
        print(
            f"{'Experiment':<18} {'Params(M)':<10} {'Dice':<10} "
            f"{'IoU':<10} {'Recall':<10} {'Epoch':<6}"
        )
        print("-" * 75)
        for r in completed:
            print(
                f"{r['display']:<18} "
                f"{r['params_m']:<10.2f} "
                f"{r['best_dice']:<10.4f} "
                f"{r['best_iou']:<10.4f} "
                f"{r['best_recall']:<10.4f} "
                f"{r['best_epoch']:<6.0f}"
            )
        print("=" * 75)
    else:
        print("\nNo completed experiments found.")

    total_time = datetime.now() - start_time
    print(f"\nTotal time: {total_time}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

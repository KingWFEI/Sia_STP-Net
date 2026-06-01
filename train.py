import gc
import os
import argparse
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm.auto import tqdm
from datetime import datetime
import csv

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
except Exception:
    binary_erosion = None
    distance_transform_edt = None

# from st_unet_v2 import build_model
# from st_deeplabv3plus import build_model
from net.sia_prompt_net_bdf import (build_model)
from loss.loss import LabelSmoothingBCE

try:
    from utils.utils import SequencePMTMDataset
except ImportError:
    raise ImportError("请确保 utils.py (含SequencePMTMDataset) 和 st_unet.py 都在当前目录下。")

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    return default_collate(batch)


def log_to_file(log_path, message):
    print(message)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def count_parameters(model):
    """计算模型的总参数量和可训练参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def log_configuration(log_path, args, net_name, device, model, loss_type):
    total_params, trainable_params = count_parameters(model)

    header = []
    header.append("=" * 60)
    header.append(f" IEEE TIM Experiment Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    header.append("=" * 60)

    header.append(f"[System Info]")
    header.append(f"  Device       : {device}")
    header.append(f"  GPU Count    : {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    if torch.cuda.is_available():
        header.append(f"  GPU Name     : {torch.cuda.get_device_name(0)}")

    header.append(f"\n[Data Settings]")
    header.append(f"  Train Dir    : {args.train_dir}")
    header.append(f"  Val Dir      : {args.val_dir}")
    header.append(f"  Window Size  : {args.window_size} Frames (T)")
    header.append(f"  Img Size     : {args.img_size}x{args.img_size}")
    header.append(f"  Use Polar    : {args.use_polar}")

    header.append(f"\n[Model Architecture]")
    header.append(f"  Net          : {net_name}")
    header.append(f"  Total Params : {total_params / 1e6:.2f} M")
    header.append(f"  Train Params : {trainable_params / 1e6:.2f} M")

    header.append(f"\n[Training Hyperparameters]")
    header.append(f"  Epochs       : {args.epochs}")
    header.append(f"  Batch Size   : {args.batch_size}")
    header.append(
        f"  Accumulation : {args.accumulation_steps} (Effective BS = {args.batch_size * args.accumulation_steps})")
    header.append(f"  Learning Rate: {args.lr}")
    header.append(f"  Weight Decay : {args.weight_decay}")
    header.append(f"  LossType   : {loss_type}")

    header.append("=" * 60 + "\n")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header))
    print("\n".join(header))


class ImprovedEarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.best_state_dict = None
        self.early_stop = False

    def __call__(self, metrics, epoch, model):
        dice = metrics['dice']
        iou = metrics['iou']
        composite_score = 0.6 * dice + 0.4 * iou

        if self.best_score is None:
            self.best_score = composite_score
            self.best_epoch = epoch
            if self.restore_best:
                self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False

        if composite_score > self.best_score + self.min_delta:
            self.best_score = composite_score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best:
                self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n[EarlyStopping] Triggered at epoch {epoch}!")
        return self.early_stop


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, warmup_start_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * \
                 (self.current_epoch / self.warmup_epochs)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / \
                       (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * \
                 0.5 * (1 + np.cos(np.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_last_lr(self):
        return [self.optimizer.param_groups[0]['lr']]


class TrainingVisualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.metrics = {
            'train_loss': [], 'val_loss': [],
            'train_dice': [], 'val_dice': [],
            'train_iou': [], 'val_iou': [],
            'train_recall': [], 'val_recall': [],
            'train_precision': [], 'val_precision': [],
            'lr': []
        }

        # 初始化 CSV 日志文件
        self.csv_path = os.path.join(save_dir, "history.csv")
        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'LR', 'Train_Loss', 'Val_Loss', 'Train_Dice', 'Val_Dice',
                             'Train_IoU', 'Val_IoU', 'Train_Recall', 'Val_Recall',
                             'Train_Precision', 'Val_Precision', 'Train_Specificity', 'Val_Specificity'])

    def update(self, epoch, train_metrics, val_metrics, lr):
        # 更新绘图字典
        self.metrics['train_loss'].append(train_metrics['loss'])
        self.metrics['val_loss'].append(val_metrics['loss'])
        self.metrics['train_dice'].append(train_metrics['dice'])
        self.metrics['val_dice'].append(val_metrics['dice'])
        self.metrics['train_iou'].append(train_metrics['iou'])
        self.metrics['val_iou'].append(val_metrics['iou'])
        self.metrics['train_recall'].append(train_metrics['recall'])
        self.metrics['val_recall'].append(val_metrics['recall'])
        self.metrics['train_precision'].append(train_metrics['precision'])
        self.metrics['val_precision'].append(val_metrics['precision'])
        self.metrics['lr'].append(lr)

        # 写入 CSV (方便后续论文作图)
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, lr, train_metrics['loss'], val_metrics['loss'],
                train_metrics['dice'], val_metrics['dice'],
                train_metrics['iou'], val_metrics['iou'],
                train_metrics['recall'], val_metrics['recall'],
                train_metrics['precision'], val_metrics['precision'],
                train_metrics['specificity'], val_metrics['specificity']
            ])

    def plot(self, best_epoch=None):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        epochs = range(1, len(self.metrics['train_loss']) + 1)

        # 1. Loss
        axes[0].plot(epochs, self.metrics['train_loss'], 'b-', label='Train')
        axes[0].plot(epochs, self.metrics['val_loss'], 'r-', label='Val')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # 2. Dice
        axes[1].plot(epochs, self.metrics['train_dice'], 'b-', label='Train')
        axes[1].plot(epochs, self.metrics['val_dice'], 'r-', label='Val')
        if best_epoch: axes[1].axvline(x=best_epoch, color='g', linestyle='--', label='Best')
        axes[1].set_title('Dice Score')
        axes[1].legend()
        axes[1].grid(True)

        # 3. IoU
        axes[2].plot(epochs, self.metrics['train_iou'], 'b-', label='Train')
        axes[2].plot(epochs, self.metrics['val_iou'], 'r-', label='Val')
        axes[2].set_title('IoU Score')
        axes[2].legend()
        axes[2].grid(True)

        # 4. Recall
        axes[3].plot(epochs, self.metrics['train_recall'], 'b-', label='Train')
        axes[3].plot(epochs, self.metrics['val_recall'], 'r-', label='Val')
        axes[3].set_title('Recall (Sensitivity)')
        axes[3].legend()
        axes[3].grid(True)

        # 5. Precision
        axes[4].plot(epochs, self.metrics['train_precision'], 'b-', label='Train')
        axes[4].plot(epochs, self.metrics['val_precision'], 'r-', label='Val')
        axes[4].set_title('Precision')
        axes[4].legend()
        axes[4].grid(True)

        # 6. LR
        axes[5].plot(epochs, self.metrics['lr'], 'purple')
        axes[5].set_title('Learning Rate')
        axes[5].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'comprehensive_training_curves.png'))
        plt.close()


def nan_value():
    return float("nan")


def as_float(value, default=0.0):
    try:
        if torch.is_tensor(value):
            value = value.detach().float().mean().item()
        return float(value)
    except Exception:
        return default


def fmt_metric(value, digits=4):
    if value is None:
        return "N/A"
    try:
        value = float(value)
        if math.isnan(value):
            return "N/A"
        return f"{value:.{digits}f}"
    except Exception:
        return "N/A"


def safe_hd95(pred_mask, target_mask):
    pred = np.asarray(pred_mask).astype(bool)
    target = np.asarray(target_mask).astype(bool)
    if pred.shape != target.shape:
        return nan_value()
    diag = float(math.hypot(pred.shape[-2], pred.shape[-1]))
    if not pred.any() and not target.any():
        return 0.0
    if pred.any() != target.any():
        return diag
    if binary_erosion is None or distance_transform_edt is None:
        return nan_value()
    try:
        pred_surface = pred ^ binary_erosion(pred)
        target_surface = target ^ binary_erosion(target)
        if not pred_surface.any() or not target_surface.any():
            return diag
        dt_pred = distance_transform_edt(~pred_surface)
        dt_target = distance_transform_edt(~target_surface)
        distances = np.concatenate([dt_target[pred_surface], dt_pred[target_surface]])
        return float(np.percentile(distances, 95)) if distances.size else 0.0
    except Exception:
        return nan_value()


def compute_batch_hd95(logits, targets, threshold=0.5):
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    preds = (torch.sigmoid(logits) > threshold).detach().cpu().numpy()
    masks = (targets > 0.5).detach().cpu().numpy()
    values = []
    for pred, mask in zip(preds, masks):
        values.append(safe_hd95(pred.squeeze(), mask.squeeze()))
    valid = [v for v in values if not math.isnan(v)]
    return float(np.mean(valid)) if valid else nan_value()


def collect_output_diagnostics(outputs):
    stats = {
        "gate_mean": nan_value(),
        "gate_std": nan_value(),
        "gate_min": nan_value(),
        "gate_max": nan_value(),
        "prompt_norm": nan_value(),
        "target_feat_norm": nan_value(),
        "merged_state_norm": nan_value(),
        "prompt_target_ratio": nan_value(),
    }
    if not isinstance(outputs, dict):
        return stats

    gate_maps = outputs.get("gate_maps")
    if torch.is_tensor(gate_maps):
        gate = gate_maps.detach().float()
        stats["gate_mean"] = as_float(gate.mean(), nan_value())
        stats["gate_std"] = as_float(gate.std(unbiased=False), nan_value())
        stats["gate_min"] = as_float(gate.min(), nan_value())
        stats["gate_max"] = as_float(gate.max(), nan_value())

    prompt_norm = outputs.get("prompt_norm")
    target_feat_norm = outputs.get("target_feat_norm")
    merged_state_norm = outputs.get("merged_state_norm")

    if prompt_norm is None:
        prompt = outputs.get("prompt", outputs.get("temporal_prompt"))
        if torch.is_tensor(prompt):
            prompt_norm = torch.linalg.vector_norm(prompt.detach().float(), dim=1).mean()
    if target_feat_norm is None:
        target_feat = outputs.get("target_feat")
        if torch.is_tensor(target_feat):
            target_feat_norm = torch.linalg.vector_norm(target_feat.detach().float(), dim=1).mean()
    if merged_state_norm is None:
        merged_state = outputs.get("merged_state")
        if torch.is_tensor(merged_state):
            merged_state_norm = torch.linalg.vector_norm(merged_state.detach().float(), dim=1).mean()

    stats["prompt_norm"] = as_float(prompt_norm, nan_value()) if prompt_norm is not None else nan_value()
    stats["target_feat_norm"] = as_float(target_feat_norm, nan_value()) if target_feat_norm is not None else nan_value()
    stats["merged_state_norm"] = as_float(merged_state_norm, nan_value()) if merged_state_norm is not None else nan_value()
    if not math.isnan(stats["prompt_norm"]) and not math.isnan(stats["target_feat_norm"]):
        stats["prompt_target_ratio"] = stats["prompt_norm"] / (stats["target_feat_norm"] + 1e-7)
    return stats


class TrainingDiagnosticsLogger:
    def __init__(self, diagnostics_dir, args, model, device, params_m, trainable_params_m, flops_g=nan_value()):
        self.diagnostics_dir = diagnostics_dir
        os.makedirs(self.diagnostics_dir, exist_ok=True)
        self.epoch_csv = os.path.join(self.diagnostics_dir, "epoch_diagnostics.csv")
        self.eff_csv = os.path.join(self.diagnostics_dir, "efficiency_log.csv")
        self.report_path = os.path.join(self.diagnostics_dir, "defect_report.md")
        self.best_json = os.path.join(self.diagnostics_dir, "best_epoch_summary.json")
        self.config_json = os.path.join(self.diagnostics_dir, "diagnostic_config.json")
        self.rows = []
        self.params_m = params_m
        self.trainable_params_m = trainable_params_m
        self.flops_g = flops_g
        self.start_time = time.time()

        self.epoch_fields = [
            "epoch", "lr", "train_loss", "val_loss",
            "train_dice", "val_dice", "dice_gap",
            "train_iou", "val_iou", "iou_gap",
            "train_precision", "val_precision", "train_recall", "val_recall",
            "train_specificity", "val_specificity",
            "val_fpr", "val_fnr", "val_pred_fg_ratio", "val_gt_fg_ratio", "val_pred_gt_fg_ratio",
            "val_hd95", "epoch_time_sec", "train_time_sec", "val_time_sec", "samples_per_sec",
            "max_gpu_memory_mb", "grad_norm",
            "gate_mean", "gate_std", "gate_min", "gate_max",
            "prompt_norm", "target_feat_norm", "merged_state_norm", "prompt_target_ratio",
        ]
        self.eff_fields = [
            "epoch", "params_m", "trainable_params_m", "epoch_time_sec", "train_time_sec",
            "val_time_sec", "samples_per_sec", "max_gpu_memory_mb", "current_lr", "flops_g",
        ]
        self._init_csv(self.epoch_csv, self.epoch_fields)
        self._init_csv(self.eff_csv, self.eff_fields)
        self._write_config(args, device)
        self.log_efficiency(0, 0.0, 0.0, 0.0, 0.0, self._current_memory_mb(), args.lr)

    def _init_csv(self, path, fields):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    def _write_config(self, args, device):
        config = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "compute_hd95": bool(args.compute_hd95),
            "hd95_every": int(args.hd95_every),
            "train_dir": args.train_dir,
            "val_dir": args.val_dir,
            "save_dir": args.save_dir,
            "window_size": args.window_size,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "accumulation_steps": args.accumulation_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "amp": bool(args.amp),
        }
        with open(self.config_json, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def _current_memory_mb(self):
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / 1024 / 1024)
        return 0.0

    def log_epoch(self, epoch, lr, train_metrics, val_metrics, timing):
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_metrics.get("loss", nan_value()),
            "val_loss": val_metrics.get("loss", nan_value()),
            "train_dice": train_metrics.get("dice", nan_value()),
            "val_dice": val_metrics.get("dice", nan_value()),
            "dice_gap": train_metrics.get("dice", nan_value()) - val_metrics.get("dice", nan_value()),
            "train_iou": train_metrics.get("iou", nan_value()),
            "val_iou": val_metrics.get("iou", nan_value()),
            "iou_gap": train_metrics.get("iou", nan_value()) - val_metrics.get("iou", nan_value()),
            "train_precision": train_metrics.get("precision", nan_value()),
            "val_precision": val_metrics.get("precision", nan_value()),
            "train_recall": train_metrics.get("recall", nan_value()),
            "val_recall": val_metrics.get("recall", nan_value()),
            "train_specificity": train_metrics.get("specificity", nan_value()),
            "val_specificity": val_metrics.get("specificity", nan_value()),
            "val_fpr": val_metrics.get("fpr", nan_value()),
            "val_fnr": val_metrics.get("fnr", nan_value()),
            "val_pred_fg_ratio": val_metrics.get("pred_fg_ratio", nan_value()),
            "val_gt_fg_ratio": val_metrics.get("gt_fg_ratio", nan_value()),
            "val_pred_gt_fg_ratio": val_metrics.get("pred_gt_fg_ratio", nan_value()),
            "val_hd95": val_metrics.get("hd95", nan_value()),
            "epoch_time_sec": timing.get("epoch_time_sec", 0.0),
            "train_time_sec": timing.get("train_time_sec", 0.0),
            "val_time_sec": timing.get("val_time_sec", 0.0),
            "samples_per_sec": timing.get("samples_per_sec", 0.0),
            "max_gpu_memory_mb": timing.get("max_gpu_memory_mb", 0.0),
            "grad_norm": train_metrics.get("grad_norm", nan_value()),
            "gate_mean": val_metrics.get("gate_mean", nan_value()),
            "gate_std": val_metrics.get("gate_std", nan_value()),
            "gate_min": val_metrics.get("gate_min", nan_value()),
            "gate_max": val_metrics.get("gate_max", nan_value()),
            "prompt_norm": val_metrics.get("prompt_norm", nan_value()),
            "target_feat_norm": val_metrics.get("target_feat_norm", nan_value()),
            "merged_state_norm": val_metrics.get("merged_state_norm", nan_value()),
            "prompt_target_ratio": val_metrics.get("prompt_target_ratio", nan_value()),
        }
        self.rows.append(row)
        with open(self.epoch_csv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.epoch_fields).writerow(row)
        self.log_efficiency(
            epoch, row["epoch_time_sec"], row["train_time_sec"], row["val_time_sec"],
            row["samples_per_sec"], row["max_gpu_memory_mb"], lr
        )

    def log_efficiency(self, epoch, epoch_time, train_time, val_time, samples_per_sec, max_gpu_memory_mb, lr):
        row = {
            "epoch": epoch,
            "params_m": self.params_m,
            "trainable_params_m": self.trainable_params_m,
            "epoch_time_sec": epoch_time,
            "train_time_sec": train_time,
            "val_time_sec": val_time,
            "samples_per_sec": samples_per_sec,
            "max_gpu_memory_mb": max_gpu_memory_mb,
            "current_lr": lr,
            "flops_g": self.flops_g,
        }
        with open(self.eff_csv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.eff_fields).writerow(row)

    def finalize(self):
        if not self.rows:
            return
        generate_best_epoch_summary(self.rows, self.best_json, self.params_m)
        generate_diagnostic_curves(self.rows, self.diagnostics_dir)
        generate_defect_report(self.rows, self.report_path, self.params_m, self.trainable_params_m, time.time() - self.start_time)


def generate_best_epoch_summary(rows, output_path, params_m):
    best = max(rows, key=lambda r: r.get("val_dice", -1.0))
    summary = {
        "best_epoch_by_val_dice": int(best["epoch"]),
        "best_val_dice": best.get("val_dice"),
        "val_iou": best.get("val_iou"),
        "val_precision": best.get("val_precision"),
        "val_recall": best.get("val_recall"),
        "val_specificity": best.get("val_specificity"),
        "val_fpr": best.get("val_fpr"),
        "val_fnr": best.get("val_fnr"),
        "val_hd95": best.get("val_hd95"),
        "pred_fg_ratio": best.get("val_pred_fg_ratio"),
        "gt_fg_ratio": best.get("val_gt_fg_ratio"),
        "pred_gt_fg_ratio": best.get("val_pred_gt_fg_ratio"),
        "params_m": params_m,
        "max_gpu_memory_mb": best.get("max_gpu_memory_mb"),
        "epoch_time_sec": best.get("epoch_time_sec"),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def generate_diagnostic_curves(rows, diagnostics_dir):
    epochs = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    axes[0].plot(epochs, [r["dice_gap"] for r in rows], label="Dice Gap")
    axes[0].set_title("Train-Val Dice Gap")
    axes[0].grid(True)

    axes[1].plot(epochs, [r["val_precision"] for r in rows], label="Precision")
    axes[1].plot(epochs, [r["val_recall"] for r in rows], label="Recall")
    axes[1].set_title("Precision vs Recall")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(epochs, [r["val_fpr"] for r in rows], label="FPR")
    axes[2].plot(epochs, [r["val_fnr"] for r in rows], label="FNR")
    axes[2].set_title("FPR / FNR")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(epochs, [r["val_pred_gt_fg_ratio"] for r in rows], label="Pred/GT")
    axes[3].axhline(1.0, color="gray", linestyle="--")
    axes[3].set_title("Pred/GT Foreground Ratio")
    axes[3].grid(True)

    axes[4].plot(epochs, [r["max_gpu_memory_mb"] for r in rows], label="GPU MB")
    axes[4].set_title("GPU Memory")
    axes[4].grid(True)

    gate_mean = [r["gate_mean"] for r in rows]
    gate_std = [r["gate_std"] for r in rows]
    if any(not math.isnan(v) for v in gate_mean):
        axes[5].plot(epochs, gate_mean, label="Gate Mean")
        axes[5].plot(epochs, gate_std, label="Gate Std")
        axes[5].legend()
    axes[5].set_title("Gate Mean / Std")
    axes[5].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(diagnostics_dir, "diagnostic_curves.png"))
    plt.close()


def generate_defect_report(rows, report_path, params_m, trainable_params_m, total_time_sec):
    best = max(rows, key=lambda r: r.get("val_dice", -1.0))
    last = rows[-1]
    avg_epoch_time = float(np.mean([r["epoch_time_sec"] for r in rows]))
    max_memory = max(r["max_gpu_memory_mb"] for r in rows)
    hd95_values = [r["val_hd95"] for r in rows if not math.isnan(r["val_hd95"])]

    suggestions = []
    lines = ["# Training Defect Report", ""]
    lines += [
        "## 1. Basic Summary",
        f"- Best Val Dice epoch: {best['epoch']}",
        f"- Best Val Dice: {fmt_metric(best['val_dice'])}",
        f"- Val Precision / Recall / Specificity: {fmt_metric(best['val_precision'])} / {fmt_metric(best['val_recall'])} / {fmt_metric(best['val_specificity'])}",
        f"- Val FPR / FNR: {fmt_metric(best['val_fpr'])} / {fmt_metric(best['val_fnr'])}",
        f"- Pred/GT foreground ratio: {fmt_metric(best['val_pred_gt_fg_ratio'])}",
        f"- HD95: {fmt_metric(best['val_hd95'])}",
        f"- Params / Trainable Params: {params_m:.2f}M / {trainable_params_m:.2f}M",
        f"- Total training time: {total_time_sec / 60.0:.2f} min",
        f"- Average epoch time: {avg_epoch_time:.2f} sec",
        f"- Max GPU memory: {max_memory:.1f} MB",
        "",
    ]

    gap = best["dice_gap"]
    lines.append("## 2. Overfitting Diagnosis")
    if gap < 0.08:
        lines.append("训练-验证 Dice gap 较小，当前过拟合风险较低。")
    elif gap < 0.15:
        lines.append(f"最佳 epoch 的 Dice gap 为 {gap:.4f}，存在轻度过拟合迹象。")
        suggestions += ["增强数据增强", "适当增大 weight decay", "继续使用 early stopping"]
    else:
        lines.append(f"模型在 epoch {best['epoch']} 出现明显 train-validation gap，可能存在明显过拟合。")
        suggestions += ["增强数据增强", "降低模型容量", "检查训练集和验证集分布是否一致"]
    lines.append("")

    ratio = best["val_pred_gt_fg_ratio"]
    precision = best["val_precision"]
    recall = best["val_recall"]
    fpr = best["val_fpr"]
    fnr = best["val_fnr"]

    lines.append("## 3. Over-segmentation Diagnosis")
    if ratio > 1.3 and precision < recall:
        lines.append("模型存在明显过分割：预测前景比例显著高于标签，且 Precision 低于 Recall。")
        suggestions += ["增强背景约束", "增加边界约束", "尝试阈值从 0.5 调整到 0.55 或 0.6"]
    elif 1.1 <= ratio <= 1.3:
        lines.append("模型存在轻度过分割倾向，预测前景比例略高。")
    elif ratio < 0.8:
        lines.append("预测区域偏小，不是过分割主导，更可能是漏分或预测保守。")
    else:
        lines.append("预测前景比例较合理，未见明显过分割。")
    lines.append(f"当前 Precision={fmt_metric(precision)}, Recall={fmt_metric(recall)}, FPR={fmt_metric(fpr)}。")
    lines.append("")

    lines.append("## 4. Under-segmentation / Missed Detection Diagnosis")
    if recall < 0.75 and fnr > 0.25:
        lines.append("漏检明显：Recall 偏低且 FNR 偏高。")
        suggestions += ["降低可靠性门控抑制强度", "将 e_hat_t = r_t * e_t 改为 (0.5 + 0.5*r_t) * e_t", "增强弱边界样本"]
    elif recall < 0.8:
        lines.append("召回能力偏弱，模型可能漏掉部分破裂区域。")
        suggestions += ["适当提高 Dice loss 权重", "检查 pos_weight 是否过低"]
    else:
        lines.append("Recall 处于相对可接受范围，漏检不是当前最突出的风险。")
    if ratio < 0.8:
        lines.append("Pred/GT foreground ratio < 0.8，说明预测区域偏小，模型可能过于保守。")
    lines.append("")

    lines.append("## 5. Boundary Quality Diagnosis")
    if hd95_values:
        if len(hd95_values) >= 2 and best["val_dice"] > rows[0]["val_dice"] and best["val_hd95"] >= rows[0]["val_hd95"]:
            lines.append("Dice 有提升但 HD95 没有同步下降，说明区域重叠变好但边界质量没有同步改善。")
            suggestions += ["保存边界误差最大的样本", "增加局部放大可视化", "检查 mask 标注边界质量"]
        hd95_std = float(np.std(hd95_values))
        lines.append(f"HD95 均值/波动: {float(np.mean(hd95_values)):.4f} / {hd95_std:.4f}。波动越大说明边界稳定性越不足。")
        if hd95_std > max(float(np.mean(hd95_values)) * 0.3, 1e-6):
            suggestions += ["增加边界相关指标监控", "引入边界约束或后处理"]
    else:
        lines.append("本次未计算 HD95。需要边界质量诊断时请加 --compute_hd95。")
    lines.append("")

    lines.append("## 6. BDF Gate Diagnosis")
    gate_mean = best.get("gate_mean", nan_value())
    gate_std = best.get("gate_std", nan_value())
    if math.isnan(gate_mean):
        lines.append("模型输出未包含 gate_maps，跳过 BDF 门控诊断。")
    else:
        lines.append(f"gate_mean={fmt_metric(gate_mean)}, gate_std={fmt_metric(gate_std)}。")
        if gate_mean < 0.25:
            lines.append("可靠性门控可能过强，真实弱边界证据可能被压制，容易导致 Recall 偏低。")
            suggestions += ["使用残差式门控", "减弱可靠性门控抑制", "给目标帧加入 target_project", "给最终 prompt 残差增加可学习 gamma"]
        elif gate_mean > 0.85:
            lines.append("门控可能过弱或几乎不起作用，可能无法筛选不可靠跨帧响应。")
        elif 0.35 <= gate_mean <= 0.75:
            lines.append("门控强度相对合理。")
        else:
            lines.append("门控强度处于中间过渡区，需要结合 Recall 和 Pred/GT 比例判断。")
        if not math.isnan(gate_std) and gate_std < 0.05:
            lines.append("gate_std 很低，门控缺乏空间区分性。")
    lines.append("")

    lines.append("## 7. Efficiency Diagnosis")
    lines.append(f"参数量 {params_m:.2f}M，平均 epoch 时间 {avg_epoch_time:.2f}s，最后 epoch 吞吐 {fmt_metric(last['samples_per_sec'])} samples/s。")
    lines.append(f"最大显存占用 {max_memory:.1f} MB。")
    if max_memory > 14 * 1024:
        lines.append("显存占用接近 RTX 5060 Ti 16GB 上限，建议降低 batch size 或开启 checkpointing。")
        suggestions += ["降低 batch size", "开启 gradient checkpointing", "检查 Transformer 或时序模块显存占用"]
    lines.append("")

    lines.append("## 8. Suggested Improvements")
    if not suggestions:
        suggestions = ["当前诊断未发现单一突出缺陷，建议优先查看预测可视化和边界误差样本。"]
    for item in dict.fromkeys(suggestions):
        lines.append(f"- {item}")
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, eps=1e-7):
        pred = pred.sigmoid()
        num = 2 * (pred * target).sum(dim=(2, 3))
        den = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 1 - (num + eps) / (den + eps)
        return dice.mean()

class BaselineLossWrapper(nn.Module):
    """
    用来包装你原本的 LabelSmoothingBCE 和 DiceLoss。
    这样一来，train_one_epoch 依然能保持极简，不需要再写拆包循环。
    """

    def __init__(self, pos_weight, smoothing=0.1):
        super().__init__()
        self.criterion_bce = LabelSmoothingBCE(pos_weight=pos_weight, smoothing=smoothing)
        self.criterion_dice = DiceLoss()
        self.aux_weights = [0.4, 0.2, 0.1]

    def _calc_single(self, pred, target):
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
        l_bce = self.criterion_bce(pred, target)
        l_dice = self.criterion_dice(pred, target)
        return 0.5 * l_bce + 0.5 * l_dice

    def forward(self, outputs, targets):
        if not isinstance(outputs, dict):
            return self._calc_single(outputs, targets)

        # 提取主干预测
        loss = self._calc_single(outputs['seg'], targets)

        # 提取辅助头预测 (Deep Supervision)
        if 'aux' in outputs:
            for i, aux_pred in enumerate(outputs['aux']):
                w = self.aux_weights[i] if i < len(self.aux_weights) else 0.1
                loss += w * self._calc_single(aux_pred, targets)

        return loss

def compute_all_metrics(logits, targets, threshold=0.5, eps=1e-7):
    """
    全量论文指标计算 (Dice, IoU, Recall, Precision, Specificity)
    """
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    recall = (tp + eps) / (tp + fn + eps)  # Sensitivity (敏感度)
    precision = (tp + eps) / (tp + fp + eps)  # PPV
    specificity = (tn + eps) / (tn + fp + eps)  # 特异度 (防误诊)
    fpr = fp / (fp + tn + eps)
    fnr = fn / (fn + tp + eps)
    pred_fg_ratio = preds.sum() / (preds.numel() + eps)
    gt_fg_ratio = targets.sum() / (targets.numel() + eps)
    pred_gt_fg_ratio = pred_fg_ratio / (gt_fg_ratio + eps)

    return {
        "dice": float(dice.item()),
        "iou": float(iou.item()),
        "recall": float(recall.item()),
        "precision": float(precision.item()),
        "specificity": float(specificity.item()),
        "fpr": float(fpr.item()),
        "fnr": float(fnr.item()),
        "pred_fg_ratio": float(pred_fg_ratio.item()),
        "gt_fg_ratio": float(gt_fg_ratio.item()),
        "pred_gt_fg_ratio": float(pred_gt_fg_ratio.item()),
        "TP": float(tp.item()),
        "FP": float(fp.item()),
        "FN": float(fn.item()),
        "TN": float(tn.item()),
    }


def train_one_epoch(model, dataloader, optimizer, criterion,
                    device, epoch, total_epochs, scaler=None, use_amp=False,
                    accumulation_steps=1):
    model.train()
    total_loss = 0.0
    metrics_keys = [
        "dice", "iou", "recall", "precision", "specificity",
        "fpr", "fnr", "pred_fg_ratio", "gt_fg_ratio", "pred_gt_fg_ratio",
        "TP", "FP", "FN", "TN",
    ]
    total_metrics = {k: 0.0 for k in metrics_keys}
    n_batches = len(dataloader)
    grad_norm_total = 0.0
    grad_norm_steps = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Train [{epoch}/{total_epochs}]", ncols=110)

    for i, batch in enumerate(pbar):
        if batch is None: continue

        seq_imgs, masks = batch
        seq_imgs = seq_imgs.to(device)
        masks = masks.to(device)

        with autocast('cuda', enabled=use_amp):
            outputs = model(seq_imgs)

            loss = criterion(outputs, masks)
            loss /= accumulation_steps

            # 提取主输出用于计算度量指标
            pred_for_metric = outputs['seg'] if isinstance(outputs, dict) else outputs

        if use_amp and scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == n_batches:
            if use_amp and scaler:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            grad_norm_total += as_float(grad_norm, 0.0)
            grad_norm_steps += 1
            optimizer.zero_grad()

        current_loss_val = loss.item() * accumulation_steps
        total_loss += current_loss_val

        with torch.no_grad():
            if pred_for_metric.shape[-2:] != masks.shape[-2:]:
                pred_for_metric = F.interpolate(pred_for_metric, size=masks.shape[-2:], mode='bilinear',
                                                align_corners=False)
            m = compute_all_metrics(pred_for_metric.detach(), masks)
            for k in total_metrics: total_metrics[k] += m[k]

        pbar.set_postfix(loss=f"{current_loss_val:.3f}", d=f"{m['dice']:.3f}", r=f"{m['recall']:.3f}")

    avg_loss = total_loss / n_batches
    for k in total_metrics: total_metrics[k] /= n_batches
    total_metrics["grad_norm"] = grad_norm_total / max(grad_norm_steps, 1)
    return avg_loss, total_metrics


def validate_one_epoch(model, dataloader,criterion,device,
                       epoch, total_epochs, use_amp=False,
                       compute_hd95=False, hd95_threshold=0.5):
    model.eval()
    old_return_diagnostics = getattr(model, "return_diagnostics", False)
    if hasattr(model, "return_diagnostics"):
        model.return_diagnostics = True
    total_loss = 0.0
    metrics_keys = [
        "dice", "iou", "recall", "precision", "specificity",
        "fpr", "fnr", "pred_fg_ratio", "gt_fg_ratio", "pred_gt_fg_ratio",
        "TP", "FP", "FN", "TN",
    ]
    total_metrics = {k: 0.0 for k in metrics_keys}
    optional_keys = [
        "gate_mean", "gate_std", "gate_min", "gate_max",
        "prompt_norm", "target_feat_norm", "merged_state_norm", "prompt_target_ratio",
    ]
    optional_totals = {k: 0.0 for k in optional_keys}
    optional_counts = {k: 0 for k in optional_keys}
    hd95_values = []
    n_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f"Val   [{epoch}/{total_epochs}]", ncols=110)
    try:
        with torch.no_grad():
            for batch in pbar:
                if batch is None: continue

                seq_imgs, masks = batch
                seq_imgs = seq_imgs.to(device)
                masks = masks.to(device)

                with autocast('cuda', enabled=use_amp):
                    outputs = model(seq_imgs)
                    loss = criterion(outputs, masks)

                pred_for_metric = outputs['seg'] if isinstance(outputs, dict) else outputs
                total_loss += loss.item()

                if pred_for_metric.shape[-2:] != masks.shape[-2:]:
                    pred_for_metric = F.interpolate(pred_for_metric, size=masks.shape[-2:], mode='bilinear',
                                                    align_corners=False)

                m = compute_all_metrics(pred_for_metric, masks, threshold=0.5)
                for k in total_metrics: total_metrics[k] += m[k]
                opt_stats = collect_output_diagnostics(outputs)
                for k, v in opt_stats.items():
                    if not math.isnan(v):
                        optional_totals[k] += v
                        optional_counts[k] += 1
                if compute_hd95:
                    hd95 = compute_batch_hd95(pred_for_metric, masks, threshold=hd95_threshold)
                    if not math.isnan(hd95):
                        hd95_values.append(hd95)

                pbar.set_postfix(loss=f"{loss.item():.3f}", d=f"{m['dice']:.3f}", r=f"{m['recall']:.3f}")
    finally:
        if hasattr(model, "return_diagnostics"):
            model.return_diagnostics = old_return_diagnostics

    if n_batches > 0:
        avg_loss = total_loss / n_batches
        for k in total_metrics: total_metrics[k] /= n_batches
    else:
        avg_loss, total_metrics = 0.0, {k: 0.0 for k in metrics_keys}
    for k in optional_keys:
        total_metrics[k] = optional_totals[k] / optional_counts[k] if optional_counts[k] > 0 else nan_value()
    total_metrics["hd95"] = float(np.mean(hd95_values)) if hd95_values else nan_value()

    return avg_loss, total_metrics


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "train_log_v1.txt")

    print(f"Using device: {device}")

    train_dataset = SequencePMTMDataset(root_dir=args.train_dir, window_size=args.window_size,
                                        img_size=args.img_size, use_polar=args.use_polar, is_train=True)
    val_dataset = SequencePMTMDataset(root_dir=args.val_dir, window_size=args.window_size,
                                      img_size=args.img_size, use_polar=args.use_polar, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=safe_collate, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=safe_collate, num_workers=args.num_workers, pin_memory=True)

    print(f">>> Building {args.net_name} (Window={args.window_size})...")
    model = build_model(
        window_size=args.window_size,
        image_size=(args.img_size, args.img_size),
        num_classes=1,
        input_channels=1
    ).to(device)

    should_freeze = (args.pretrained is not None) or (args.resume is not None)
    if should_freeze:
        print(">>> [Info] Pretrained weights detected. Freezing Shared Encoder for first 50 epochs...")
        for param in model.parameters(): param.requires_grad = False
        for name, module in model.named_modules():
            if any(k in name for k in ["st_fusion", "aligner", "conv0_1", "conv0_2", "conv0_3", "conv0_4", "final"]):
                for param in module.parameters(): param.requires_grad = True
    else:
        print(">>> [Info] Training from scratch. All layers are trainable.")



    use_amp = (device.type == "cuda") and args.amp
    scaler = GradScaler('cuda', enabled=use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pos_w = torch.tensor([args.pos_weight if args.pos_weight else 1.0], device=device)
    loss_type = 'DiceLoss+LabelSmoothingBCE(aux=[0.4,0.2,0.1])'
    print(f">>> [Info] Using Baseline Loss ({loss_type})")
    criterion = BaselineLossWrapper(pos_weight=pos_w, smoothing=0.1).to(device)

    log_configuration(log_path, args, f' {args.net_name}', device, model, loss_type)
    if args.resume or args.pretrained:
        cp = args.resume if args.resume else args.pretrained
        if os.path.exists(cp):
            checkpoint = torch.load(cp, map_location=device)
            sd = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            model.load_state_dict(sd, strict=False)
            log_to_file(log_path, f"Loaded weights from {cp}")

    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, total_epochs=args.epochs, min_lr=1e-6)
    early_stopper = ImprovedEarlyStopping(patience=args.early_stopping)
    visualizer = TrainingVisualizer(args.save_dir)
    total_params, trainable_params = count_parameters(model)
    diagnostics_logger = None
    if args.enable_diagnostics:
        diagnostics_dir = args.diagnostics_dir or os.path.join(args.save_dir, "diagnostics")
        diagnostics_logger = TrainingDiagnosticsLogger(
            diagnostics_dir=diagnostics_dir,
            args=args,
            model=model,
            device=device,
            params_m=total_params / 1e6,
            trainable_params_m=trainable_params / 1e6,
            flops_g=nan_value(),
        )
        if args.compute_hd95 and (binary_erosion is None or distance_transform_edt is None):
            log_to_file(log_path, "[Diagnostics] scipy is unavailable; HD95 will be written as N/A.")

    # 记录最佳权重的变量
    best_val_dice = -1.0
    best_val_recall = -1.0

    try:
        for epoch in range(1, args.epochs + 1):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            epoch_start = time.time()
            if should_freeze and epoch == 51:
                log_to_file(log_path, f"\n>>> [Epoch {epoch}] Unfreezing All Layers... Start Fine-tuning! <<<")
                for param in model.parameters(): param.requires_grad = True

            train_start = time.time()
            t_loss, t_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion,
                device, epoch, args.epochs, scaler, use_amp, accumulation_steps=args.accumulation_steps
            )
            train_time = time.time() - train_start

            val_start = time.time()
            should_compute_hd95 = bool(args.compute_hd95) and (epoch % max(args.hd95_every, 1) == 0)
            v_loss, v_metrics = validate_one_epoch(
                model, val_loader, criterion, device,
                epoch, args.epochs, use_amp,
                compute_hd95=should_compute_hd95
            )
            val_time = time.time() - val_start
            epoch_time = time.time() - epoch_start

            t_metrics['loss'] = t_loss
            v_metrics['loss'] = v_loss

            scheduler.step()
            curr_lr = scheduler.get_last_lr()[0]
            visualizer.update(epoch, t_metrics, v_metrics, curr_lr)

            samples_per_sec = len(train_dataset) / max(train_time, 1e-7)
            max_gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
            timing = {
                "epoch_time_sec": epoch_time,
                "train_time_sec": train_time,
                "val_time_sec": val_time,
                "samples_per_sec": samples_per_sec,
                "max_gpu_memory_mb": max_gpu_memory_mb,
            }
            if diagnostics_logger:
                diagnostics_logger.log_epoch(epoch, curr_lr, t_metrics, v_metrics, timing)

            bdf_line = ""
            if not math.isnan(v_metrics.get("gate_mean", nan_value())):
                bdf_line = (
                    f"\n  BDF   - gate_mean: {fmt_metric(v_metrics.get('gate_mean'))} | "
                    f"gate_std: {fmt_metric(v_metrics.get('gate_std'))}"
                )
            log_msg = (
                f"Epoch {epoch}/{args.epochs} (LR: {curr_lr:.6f}):\n"
                f"  Train - Loss: {t_loss:.4f} | Dice: {t_metrics['dice']:.4f} | IoU: {t_metrics['iou']:.4f} | "
                f"Precision: {t_metrics['precision']:.4f} | Recall: {t_metrics['recall']:.4f} | Spec: {t_metrics['specificity']:.4f}\n"
                f"  Val   - Loss: {v_loss:.4f} | Dice: {v_metrics['dice']:.4f} | IoU: {v_metrics['iou']:.4f} | "
                f"Precision: {v_metrics['precision']:.4f} | Recall: {v_metrics['recall']:.4f} | Spec: {v_metrics['specificity']:.4f} | "
                f"FPR: {fmt_metric(v_metrics.get('fpr'))} | FNR: {fmt_metric(v_metrics.get('fnr'))} | "
                f"Pred/GT: {fmt_metric(v_metrics.get('pred_gt_fg_ratio'))}\n"
                f"  Diag  - Gap(Dice): {fmt_metric(t_metrics['dice'] - v_metrics['dice'])} | "
                f"HD95: {fmt_metric(v_metrics.get('hd95'))} | GPU: {max_gpu_memory_mb:.1f} MB | Time: {epoch_time:.1f} s"
                f"{bdf_line}"
            )
            log_to_file(log_path, log_msg)

            torch.save(model.state_dict(), os.path.join(args.save_dir, "last.pth"))

            # 1. 保存最高 Dice 权重 (总体分割最佳)
            if v_metrics['dice'] > best_val_dice:
                best_val_dice = v_metrics['dice']
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best_dice.pth"))
                log_to_file(log_path, f"  [!] New Best Dice Saved: {best_val_dice:.4f}")

            # 2. 保存最高 Recall 权重 (医学早期筛查极具价值)
            if v_metrics['recall'] > best_val_recall:
                best_val_recall = v_metrics['recall']
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best_recall.pth"))
                log_to_file(log_path, f"  [!] New Best Recall Saved: {best_val_recall:.4f}")

            if early_stopper(v_metrics, epoch, model):
                log_to_file(log_path, f"Early stopping at epoch {epoch}")
                break

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        visualizer.plot(best_epoch=early_stopper.best_epoch)
        if diagnostics_logger:
            diagnostics_logger.finalize()
        log_to_file(log_path, "Training Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default=r"D:\datasets\pmtm\TBUT_Seg_Data_v1\train")
    parser.add_argument("--val_dir", type=str, default=r"D:\datasets\pmtm\TBUT_Seg_Data_v1\val")
    parser.add_argument("--save_dir", type=str, default="./runs/st_unet_exp")

    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--use_polar", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=8)

    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pos_weight", type=float, default=2.0)
    parser.add_argument("--early_stopping", type=int, default=30)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--enable_diagnostics", action="store_true", default=True)
    parser.add_argument("--compute_hd95", action="store_true", default=False)
    parser.add_argument("--hd95_every", type=int, default=1)
    parser.add_argument("--diagnostics_dir", type=str, default=None)

    parser.add_argument("--net_name", type=str, default='xxx')

    args = parser.parse_args()
    main(args)

import gc
import os
import argparse
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

# from st_unet_v2 import build_model
# from st_deeplabv3plus import build_model
from net.sia_prompt_net_bdf import (build_model)
from loss.loss import LabelSmoothingBCE

try:
    from utils import SequencePMTMDataset
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
            for aux_pred in outputs['aux']:
                loss += 0.5 * self._calc_single(aux_pred, targets)

        return loss

def compute_all_metrics(logits, targets, threshold=0.5, eps=1e-7):
    """
    全量论文指标计算 (Dice, IoU, Recall, Precision, Specificity)
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    recall = (tp + eps) / (tp + fn + eps)  # Sensitivity (敏感度)
    precision = (tp + eps) / (tp + fp + eps)  # PPV
    specificity = (tn + eps) / (tn + fp + eps)  # 特异度 (防误诊)

    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "recall": recall.item(),
        "precision": precision.item(),
        "specificity": specificity.item()
    }


def train_one_epoch(model, dataloader, optimizer, criterion,
                    device, epoch, total_epochs, scaler=None, use_amp=False,
                    accumulation_steps=1):
    model.train()
    total_loss = 0.0
    metrics_keys = ["dice", "iou", "recall", "precision", "specificity"]
    total_metrics = {k: 0.0 for k in metrics_keys}
    n_batches = len(dataloader)
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
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
    return avg_loss, total_metrics


def validate_one_epoch(model, dataloader,criterion,device,
                       epoch, total_epochs, use_amp=False):
    model.eval()
    total_loss = 0.0
    metrics_keys = ["dice", "iou", "recall", "precision", "specificity"]
    total_metrics = {k: 0.0 for k in metrics_keys}
    n_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f"Val   [{epoch}/{total_epochs}]", ncols=110)
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

            pbar.set_postfix(loss=f"{loss.item():.3f}", d=f"{m['dice']:.3f}", r=f"{m['recall']:.3f}")

    if n_batches > 0:
        avg_loss = total_loss / n_batches
        for k in total_metrics: total_metrics[k] /= n_batches
    else:
        avg_loss, total_metrics = 0.0, {k: 0.0 for k in metrics_keys}

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
    loss_type = 'DiceLoss+LabelSmoothingBCE'
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

    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, total_epochs=args.epochs, min_lr=1e-5)
    early_stopper = ImprovedEarlyStopping(patience=args.early_stopping)
    visualizer = TrainingVisualizer(args.save_dir)

    # 记录最佳权重的变量
    best_val_dice = -1.0
    best_val_recall = -1.0

    for epoch in range(1, args.epochs + 1):
        if should_freeze and epoch == 51:
            log_to_file(log_path, f"\n>>> [Epoch {epoch}] Unfreezing All Layers... Start Fine-tuning! <<<")
            for param in model.parameters(): param.requires_grad = True

        t_loss, t_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, args.epochs, scaler, use_amp, accumulation_steps=args.accumulation_steps
        )

        v_loss, v_metrics = validate_one_epoch(
            model, val_loader, criterion, device,
            epoch, args.epochs, use_amp
        )

        t_metrics['loss'] = t_loss
        v_metrics['loss'] = v_loss

        scheduler.step()
        curr_lr = scheduler.get_last_lr()[0]
        visualizer.update(epoch, t_metrics, v_metrics, curr_lr)

        # 日志写入
        log_msg = (f"Epoch {epoch}/{args.epochs} (LR: {curr_lr:.6f}):\n"
                   f"  Train - Loss: {t_loss:.4f} | Dice: {t_metrics['dice']:.4f} | Recall: {t_metrics['recall']:.4f} | Spec: {t_metrics['specificity']:.4f}\n"
                   f"  Val   - Loss: {v_loss:.4f} | Dice: {v_metrics['dice']:.4f} | Recall: {v_metrics['recall']:.4f} | Spec: {v_metrics['specificity']:.4f}")
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
        torch.cuda.empty_cache()

    visualizer.plot(best_epoch=early_stopper.best_epoch)
    log_to_file(log_path, "Training Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default=r"D:\datasets\pmtm\TBUT_Seg_Data_v1\train")
    parser.add_argument("--val_dir", type=str, default=r"D:\datasets\pmtm\TBUT_Seg_Data_v1\val")
    parser.add_argument("--save_dir", type=str, default="./runs/st_unet_exp")

    parser.add_argument("--window_size", type=int, default=3)
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

    parser.add_argument("--net_name", type=str, default='xxx')

    args = parser.parse_args()
    main(args)

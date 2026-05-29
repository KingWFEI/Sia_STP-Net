import argparse
import csv
import gc
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm.auto import tqdm

from net.comparison_models import COMPARISON_MODELS, build_comparison_model
from loss.loss import BaselineLossWrapper
from utils.utils import SequencePMTMDataset


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"true", "1", "yes", "y"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, warmup_start_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.total_epochs = max(int(total_epochs), 1)
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.warmup_epochs > 0 and self.current_epoch <= self.warmup_epochs:
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * (
                self.current_epoch / self.warmup_epochs
            )
        else:
            denom = max(self.total_epochs - self.warmup_epochs, 1)
            progress = min(max((self.current_epoch - self.warmup_epochs) / denom, 0.0), 1.0)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_last_lr(self):
        return [self.optimizer.param_groups[0]["lr"]]


class EarlyStopping:
    def __init__(self, patience=30, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False

    def __call__(self, metrics, epoch):
        score = 0.6 * metrics["dice"] + 0.4 * metrics["iou"]
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            self.early_stop = self.counter >= self.patience
        return self.early_stop


def compute_all_metrics(logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    recall = (tp + eps) / (tp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "recall": recall.item(),
        "precision": precision.item(),
        "specificity": specificity.item(),
    }


class HistoryWriter:
    def __init__(self, save_dir):
        self.csv_path = os.path.join(save_dir, "history.csv")
        with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Epoch",
                "LR",
                "Train_Loss",
                "Val_Loss",
                "Train_Dice",
                "Val_Dice",
                "Train_IoU",
                "Val_IoU",
                "Train_Recall",
                "Val_Recall",
                "Train_Precision",
                "Val_Precision",
                "Train_Specificity",
                "Val_Specificity",
            ])

    def update(self, epoch, lr, train_metrics, val_metrics):
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                lr,
                train_metrics["loss"],
                val_metrics["loss"],
                train_metrics["dice"],
                val_metrics["dice"],
                train_metrics["iou"],
                val_metrics["iou"],
                train_metrics["recall"],
                val_metrics["recall"],
                train_metrics["precision"],
                val_metrics["precision"],
                train_metrics["specificity"],
                val_metrics["specificity"],
            ])


def log_line(log_path, message):
    print(message)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, total_epochs, scaler, use_amp, accumulation_steps):
    model.train()
    total_loss = 0.0
    metrics_keys = ["dice", "iou", "recall", "precision", "specificity"]
    total_metrics = {k: 0.0 for k in metrics_keys}
    n_batches = len(dataloader)
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Train [{epoch}/{total_epochs}]", ncols=110)
    for i, batch in enumerate(pbar):
        if batch is None:
            continue
        seq_imgs, masks = batch
        seq_imgs = seq_imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast("cuda", enabled=use_amp):
            outputs = model(seq_imgs)
            loss = criterion(outputs, masks) / accumulation_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == n_batches:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        loss_value = loss.item() * accumulation_steps
        total_loss += loss_value
        with torch.no_grad():
            pred = outputs["seg"] if isinstance(outputs, dict) else outputs
            if pred.shape[-2:] != masks.shape[-2:]:
                pred = F.interpolate(pred, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            m = compute_all_metrics(pred.detach(), masks)
            for key in metrics_keys:
                total_metrics[key] += m[key]
        pbar.set_postfix(loss=f"{loss_value:.3f}", dice=f"{m['dice']:.3f}", recall=f"{m['recall']:.3f}")

    denom = max(n_batches, 1)
    total_metrics = {k: v / denom for k, v in total_metrics.items()}
    return total_loss / denom, total_metrics


def validate_one_epoch(model, dataloader, criterion, device, epoch, total_epochs, use_amp):
    model.eval()
    total_loss = 0.0
    metrics_keys = ["dice", "iou", "recall", "precision", "specificity"]
    total_metrics = {k: 0.0 for k in metrics_keys}
    n_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f"Val   [{epoch}/{total_epochs}]", ncols=110)
    with torch.no_grad():
        for batch in pbar:
            if batch is None:
                continue
            seq_imgs, masks = batch
            seq_imgs = seq_imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                outputs = model(seq_imgs)
                loss = criterion(outputs, masks)

            pred = outputs["seg"] if isinstance(outputs, dict) else outputs
            if pred.shape[-2:] != masks.shape[-2:]:
                pred = F.interpolate(pred, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            total_loss += loss.item()
            m = compute_all_metrics(pred, masks)
            for key in metrics_keys:
                total_metrics[key] += m[key]
            pbar.set_postfix(loss=f"{loss.item():.3f}", dice=f"{m['dice']:.3f}", recall=f"{m['recall']:.3f}")

    denom = max(n_batches, 1)
    total_metrics = {k: v / denom for k, v in total_metrics.items()}
    return total_loss / denom, total_metrics


def run_single_model(args, model_name, train_loader, val_loader, device):
    model_save_dir = os.path.join(args.save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    log_path = os.path.join(model_save_dir, "training_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Comparison experiment started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    model = build_comparison_model(
        model_name=model_name,
        window_size=args.window_size,
        image_size=(args.img_size, args.img_size),
        num_classes=1,
        input_channels=1,
        input_mode=args.input_mode,
        base_ch=args.base_ch,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    log_line(log_path, "=" * 80)
    log_line(log_path, f"Model: {model_name}")
    log_line(log_path, f"Device: {device}")
    log_line(log_path, f"Input mode: {args.input_mode} | Window: {args.window_size} | Image: {args.img_size}")
    log_line(log_path, f"Params: {total_params / 1e6:.2f}M | Trainable: {trainable_params / 1e6:.2f}M")
    log_line(log_path, f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr} | WD: {args.weight_decay}")
    log_line(log_path, "=" * 80)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr=args.min_lr,
    )
    pos_weight = torch.tensor([args.pos_weight], device=device)
    criterion = BaselineLossWrapper(pos_weight=pos_weight, smoothing=args.label_smoothing).to(device)
    use_amp = (device.type == "cuda") and args.amp
    scaler = GradScaler("cuda", enabled=use_amp)
    early_stopper = EarlyStopping(patience=args.early_stopping)
    history = HistoryWriter(model_save_dir)

    best = {
        "model": model_name,
        "best_epoch": 0,
        "best_dice": -1.0,
        "best_iou": 0.0,
        "best_recall": 0.0,
        "best_precision": 0.0,
        "best_specificity": 0.0,
        "params_m": total_params / 1e6,
    }

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            args.epochs,
            scaler,
            use_amp,
            args.accumulation_steps,
        )
        val_loss, val_metrics = validate_one_epoch(model, val_loader, criterion, device, epoch, args.epochs, use_amp)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        train_metrics["loss"] = train_loss
        val_metrics["loss"] = val_loss
        history.update(epoch, lr, train_metrics, val_metrics)

        log_line(
            log_path,
            f"Epoch {epoch:03d}/{args.epochs:03d} LR {lr:.6g} | "
            f"Train L {train_loss:.4f} D {train_metrics['dice']:.4f} IoU {train_metrics['iou']:.4f} | "
            f"Val L {val_loss:.4f} D {val_metrics['dice']:.4f} IoU {val_metrics['iou']:.4f} "
            f"R {val_metrics['recall']:.4f} P {val_metrics['precision']:.4f}",
        )

        torch.save(model.state_dict(), os.path.join(model_save_dir, "last.pth"))
        if val_metrics["dice"] > best["best_dice"]:
            best.update({
                "best_epoch": epoch,
                "best_dice": val_metrics["dice"],
                "best_iou": val_metrics["iou"],
                "best_recall": val_metrics["recall"],
                "best_precision": val_metrics["precision"],
                "best_specificity": val_metrics["specificity"],
            })
            torch.save(model.state_dict(), os.path.join(model_save_dir, "best_dice.pth"))
            log_line(log_path, f"  [best] saved best_dice.pth with Dice={best['best_dice']:.4f}")

        if early_stopper(val_metrics, epoch):
            log_line(log_path, f"Early stopping at epoch {epoch}; best epoch was {early_stopper.best_epoch}")
            break

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

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
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=safe_collate,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model_names = list(COMPARISON_MODELS) if args.model == "all" else [args.model]
    summary_rows = []
    for model_name in model_names:
        print(f"\n>>> Running comparison model: {model_name}")
        best = run_single_model(args, model_name, train_loader, val_loader, device)
        summary_rows.append(best)

    summary_path = os.path.join(args.save_dir, "comparison_summary.csv")
    with open(summary_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model",
            "best_epoch",
            "best_dice",
            "best_iou",
            "best_recall",
            "best_precision",
            "best_specificity",
            "params_m",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nComparison finished. Summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default=r"D:\datasets\pmtm\TBUT_Seg_Data_v1\train")
    parser.add_argument("--val_dir", type=str, default=r"D:\datasets\pmtm\TBUT_Seg_Data_v1\val")
    parser.add_argument("--save_dir", type=str, default="./runs/comparison_baselines")
    parser.add_argument("--model", type=str, default="all", choices=["all", *COMPARISON_MODELS])
    parser.add_argument("--input_mode", type=str, default="stack", choices=["stack", "center"])
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--use_polar", type=str2bool, default=False)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--pos_weight", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--early_stopping", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True)

    main(parser.parse_args())

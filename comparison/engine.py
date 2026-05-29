import csv
import gc
import os
import random
import time
import traceback

import cv2
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm.auto import tqdm

from comparison.losses import UnifiedSegLoss
from comparison.metrics import (
    SegmentationMetricAccumulator,
    count_parameters_m,
    estimate_flops_g,
    get_logits,
    measure_fps,
)
from comparison.model_registry import build_registered_model
from utils.utils import SequencePMTMDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def safe_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return default_collate(batch)


def make_output_dirs(output_dir):
    dirs = {
        "root": output_dir,
        "checkpoints": os.path.join(output_dir, "checkpoints"),
        "logs": os.path.join(output_dir, "logs"),
        "predictions": os.path.join(output_dir, "predictions"),
        "metrics": os.path.join(output_dir, "metrics"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def build_loaders(cfg, batch_size=None):
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    batch_size = int(batch_size or train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 4))
    pin_memory = torch.cuda.is_available()

    train_dataset = SequencePMTMDataset(
        root_dir=data_cfg["train_dir"],
        window_size=int(data_cfg["window_size"]),
        img_size=int(data_cfg["img_size"]),
        use_polar=bool(data_cfg.get("use_polar", False)),
        is_train=True,
    )
    val_dataset = SequencePMTMDataset(
        root_dir=data_cfg["val_dir"],
        window_size=int(data_cfg["window_size"]),
        img_size=int(data_cfg["img_size"]),
        use_polar=bool(data_cfg.get("use_polar", False)),
        is_train=False,
    )
    test_dataset = SequencePMTMDataset(
        root_dir=data_cfg["test_dir"],
        window_size=int(data_cfg["window_size"]),
        img_size=int(data_cfg["img_size"]),
        use_polar=bool(data_cfg.get("use_polar", False)),
        is_train=False,
    )

    generator = torch.Generator()
    generator.manual_seed(int(cfg.get("seed", 42)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=safe_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=safe_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=safe_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def unpack_batch(batch):
    if isinstance(batch, dict):
        return batch["image_seq"], batch["mask"], batch
    image_seq, mask = batch[0], batch[1]
    meta = batch[2:] if len(batch) > 2 else None
    return image_seq, mask, meta


def select_model_input(image_seq, model_type, target_idx):
    if model_type == "static":
        return image_seq[:, target_idx]
    if model_type == "temporal":
        return image_seq
    raise ValueError(f"Unsupported model type: {model_type}. Use 'static' or 'temporal'.")


class CosineScheduler:
    def __init__(self, optimizer, epochs, min_lr=1e-6):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(int(epochs), 1), eta_min=float(min_lr)
        )

    def step(self):
        self.scheduler.step()

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state):
        self.scheduler.load_state_dict(state)

    def get_last_lr(self):
        return self.scheduler.get_last_lr()


def train_one_model(model_name, cfg, dirs, device, resume=False, batch_size=None):
    model_cfg = cfg["models"][model_name]
    model_type = model_cfg["type"]
    train_cfg = cfg["training"]
    target_idx = int(cfg["data"].get("target_idx", int(cfg["data"]["window_size"]) // 2))
    train_loader, val_loader, test_loader = build_loaders(cfg, batch_size=batch_size)
    model = build_registered_model(model_name, cfg).to(device)

    criterion = UnifiedSegLoss(
        pos_weight=float(train_cfg.get("pos_weight", 2.0)),
        lambda_bce=float(train_cfg.get("lambda_bce", 1.0)),
        lambda_dice=float(train_cfg.get("lambda_dice", 1.0)),
        aux_weights=train_cfg.get("aux_weights", [0.4, 0.3, 0.2]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = CosineScheduler(optimizer, int(train_cfg["epochs"]), min_lr=float(train_cfg.get("min_lr", 1e-6)))
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    ckpt_best = os.path.join(dirs["checkpoints"], f"{model_name}_best.pth")
    ckpt_last = os.path.join(dirs["checkpoints"], f"{model_name}_last.pth")
    log_path = os.path.join(dirs["logs"], f"{model_name}.log")
    history_path = os.path.join(dirs["metrics"], f"{model_name}_history.csv")
    pred_dir = os.path.join(dirs["predictions"], model_name)
    os.makedirs(pred_dir, exist_ok=True)

    start_epoch = 1
    best_dice = -1.0
    if resume and os.path.exists(ckpt_last):
        checkpoint = torch.load(ckpt_last, map_location=device)
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state, strict=False)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_dice = float(checkpoint.get("best_dice", -1.0))

    with open(history_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["Epoch", "LR", "Train_Loss", "Val_Loss", "Val_Dice", "Val_Precision", "Val_Recall", "Val_Specificity", "Val_HD95"])

    log_line(log_path, f"Start {model_name} | type={model_type} | batch={train_loader.batch_size} | resume={resume}")
    for epoch in range(start_epoch, int(train_cfg["epochs"]) + 1):
        train_loss = _train_epoch(
            model, train_loader, criterion, optimizer, scaler, use_amp, device, model_type, target_idx,
            int(train_cfg["accumulation_steps"]), epoch, int(train_cfg["epochs"])
        )
        val_loss, val_metrics = _validate_epoch(model, val_loader, criterion, use_amp, device, model_type, target_idx)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, lr, train_loss, val_loss, val_metrics["Dice"], val_metrics["Precision"],
                val_metrics["Recall"], val_metrics["Specificity"], val_metrics["HD95"],
            ])
        log_line(
            log_path,
            f"Epoch {epoch:03d}: lr={lr:.6g} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} dice={val_metrics['Dice']:.4f}",
        )

        if val_metrics["Dice"] > best_dice:
            best_dice = val_metrics["Dice"]
            torch.save(_checkpoint(model, optimizer, scheduler, epoch, best_dice, cfg), ckpt_best)
            log_line(log_path, f"Saved best checkpoint: Dice={best_dice:.4f}")
        torch.save(_checkpoint(model, optimizer, scheduler, epoch, best_dice, cfg), ckpt_last)

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return evaluate_one_model(model_name, cfg, dirs, device, test_loader=test_loader)


def evaluate_one_model(model_name, cfg, dirs, device, test_loader=None):
    model_cfg = cfg["models"][model_name]
    model_type = model_cfg["type"]
    target_idx = int(cfg["data"].get("target_idx", int(cfg["data"]["window_size"]) // 2))
    test_loader = test_loader or build_loaders(cfg)[2]
    model = build_registered_model(model_name, cfg).to(device)
    ckpt_best = os.path.join(dirs["checkpoints"], f"{model_name}_best.pth")
    if not os.path.exists(ckpt_best):
        raise FileNotFoundError(f"Missing best checkpoint for eval: {ckpt_best}")
    checkpoint = torch.load(ckpt_best, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    model.eval()

    acc = SegmentationMetricAccumulator(threshold=float(cfg["eval"].get("threshold", 0.5)))
    pred_dir = os.path.join(dirs["predictions"], model_name)
    os.makedirs(pred_dir, exist_ok=True)
    save_predictions = int(cfg["eval"].get("save_predictions", 16))
    use_amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    saved = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Test  [{model_name}]", ncols=110)
        for batch in pbar:
            if batch is None:
                continue
            image_seq, masks, _ = unpack_batch(batch)
            image_seq = image_seq.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            x = select_model_input(image_seq, model_type, target_idx)
            with autocast(enabled=use_amp):
                outputs = model(x)
            logits = get_logits(outputs)
            acc.update(logits, masks)
            if saved < save_predictions:
                saved += _save_prediction_batch(logits, pred_dir, saved, save_predictions)

    metrics = acc.compute()
    example = make_profile_input(cfg, model_type, device)
    params_m = count_parameters_m(model)
    flops_g = estimate_flops_g(model, example, device)
    fps = measure_fps(
        model,
        example,
        device,
        warmup=int(cfg["eval"].get("fps_warmup", 5)),
        iterations=int(cfg["eval"].get("fps_iterations", 20)),
    )

    return {
        "Method": model_cfg.get("display_name", model_name),
        "Params(M)": params_m,
        "FLOPs(G)": flops_g,
        "FPS": fps,
        **metrics,
    }


def _train_epoch(model, loader, criterion, optimizer, scaler, use_amp, device, model_type, target_idx, accumulation_steps, epoch, epochs):
    model.train()
    total_loss = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"Train [{epoch}/{epochs}]", ncols=110)
    for step, batch in enumerate(pbar, start=1):
        if batch is None:
            continue
        image_seq, masks, _ = unpack_batch(batch)
        image_seq = image_seq.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        x = select_model_input(image_seq, model_type, target_idx)
        with autocast(enabled=use_amp):
            outputs = model(x)
            loss = criterion(outputs, masks) / accumulation_steps
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if step % accumulation_steps == 0 or step == len(loader):
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        value = float(loss.item() * accumulation_steps)
        total_loss += value
        n += 1
        pbar.set_postfix(loss=f"{value:.4f}")
    return total_loss / max(n, 1)


def _validate_epoch(model, loader, criterion, use_amp, device, model_type, target_idx):
    model.eval()
    acc = SegmentationMetricAccumulator()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", ncols=110):
            if batch is None:
                continue
            image_seq, masks, _ = unpack_batch(batch)
            image_seq = image_seq.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            x = select_model_input(image_seq, model_type, target_idx)
            with autocast(enabled=use_amp):
                outputs = model(x)
                loss = criterion(outputs, masks)
            logits = get_logits(outputs)
            acc.update(logits, masks)
            total_loss += float(loss.item())
            n += 1
    return total_loss / max(n, 1), acc.compute()


def _checkpoint(model, optimizer, scheduler, epoch, best_dice, cfg):
    return {
        "epoch": epoch,
        "best_dice": best_dice,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": cfg,
    }


def make_profile_input(cfg, model_type, device):
    t = int(cfg["data"]["window_size"])
    c = int(cfg["data"].get("input_channels", 1))
    size = int(cfg["data"]["img_size"])
    if model_type == "static":
        return torch.randn(1, c, size, size, device=device)
    return torch.randn(1, t, c, size, size, device=device)


def _save_prediction_batch(logits, pred_dir, start_idx, limit):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).detach().cpu().numpy().astype(np.uint8) * 255
    saved = 0
    for item in preds:
        if start_idx + saved >= limit:
            break
        path = os.path.join(pred_dir, f"pred_{start_idx + saved:04d}.png")
        cv2.imwrite(path, item.squeeze())
        saved += 1
    return saved


def log_line(path, message):
    print(message)
    with open(path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def log_exception(log_path, model_name):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{model_name}] failed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(traceback.format_exc())
        f.write("\n")

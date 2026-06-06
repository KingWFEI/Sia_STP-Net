import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from contextlib import nullcontext

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from comparison.metrics import get_logits, hd95  # noqa: E402
from comparison.model_registry import build_registered_model  # noqa: E402
from scripts.visualize_comparison_figure import overlay_mask  # noqa: E402
from utils.utils import SequencePMTMDataset  # noqa: E402


METRIC_AVERAGE_MODE = "global_confusion_hd95_image_mean"

DEFAULT_MODELS = [
    "unet",
    "unet3plus",
    "2_5d_unet",
    "siamese_biconvlstm",
    "siamese_stpnet",
]

DISPLAY_NAMES = {
    "unet": "U-Net",
    "unetpp": "U-Net++",
    "deeplabv3plus": "DeepLabV3+",
    "unet3plus": "UNet 3+",
    "attention_unet": "Attention U-Net",
    "transunet": "TransUNet",
    "swinunet": "Swin-Unet",
    "2_5d_unet": "2.5D U-Net",
    "siamese_encoder_decoder": "Siamese Encoder-Decoder",
    "siamese_biconvlstm": "Siamese + Bi-ConvLSTM",
    "siamese_stpnet": "Siamese STP-Net",
}

MODEL_ALIASES = {
    "u-net": "unet",
    "unet": "unet",
    "unet 3+": "unet3plus",
    "unet3+": "unet3plus",
    "unet3plus": "unet3plus",
    "deeplabv3+": "deeplabv3plus",
    "deeplabv3plus": "deeplabv3plus",
    "transunet": "transunet",
    "transunet": "transunet",
    "swin unet": "swinunet",
    "swin-unet": "swinunet",
    "swinunet": "swinunet",
    "2.5d u-net": "2_5d_unet",
    "2_5d_unet": "2_5d_unet",
    "siamese + bi-convlstm": "siamese_biconvlstm",
    "siamese_biconvlstm": "siamese_biconvlstm",
    "sia-stp": "siamese_stpnet",
    "sia-stp(ours)": "siamese_stpnet",
    "siamese stp-net": "siamese_stpnet",
    "siamese_stpnet": "siamese_stpnet",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase-wise evaluation for PMTM broken-area segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test_dir", default=r"D:\datasets\pmtm\TUBT_test")
    parser.add_argument("--config", default=os.path.join("configs", "compare_models.yaml"))
    parser.add_argument("--checkpoint_dir", default=os.path.join("run", "comparison", "checkpoints"))
    parser.add_argument(
        "--prediction_dir",
        default=None,
        help="Optional existing/saved prediction root. Expected layout: prediction_dir/model_name/pred_0000.png.",
    )
    parser.add_argument("--output_dir", default=os.path.join("run", "comparison", "phasewise"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--target_idx", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None, help="Debug option; evaluate only the first N samples.")
    parser.add_argument("--skip_inference", action="store_true", help="Read predictions from --prediction_dir/--output_dir without running checkpoints.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing saved phase-wise predictions.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--hard_dice_drop", type=float, default=0.10)
    return parser.parse_args()


def load_yaml_config(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def normalize_model_name(name):
    key = str(name).strip()
    if key in DISPLAY_NAMES:
        return key
    return MODEL_ALIASES.get(key.lower(), key)


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(name)


def prepare_config(args):
    cfg = load_yaml_config(args.config)
    if not cfg:
        raise FileNotFoundError(f"Cannot load config: {args.config}")
    cfg.setdefault("data", {})
    cfg["data"]["test_dir"] = args.test_dir
    if args.window_size is not None:
        cfg["data"]["window_size"] = int(args.window_size)
    if args.img_size is not None:
        cfg["data"]["img_size"] = int(args.img_size)
    if args.target_idx is not None:
        cfg["data"]["target_idx"] = int(args.target_idx)
    cfg.setdefault("eval", {})
    if args.threshold is not None:
        cfg["eval"]["threshold"] = float(args.threshold)
    else:
        cfg["eval"]["threshold"] = float(cfg["eval"].get("threshold", 0.5))
    return cfg


def patient_id_from_sample(sample):
    case = str(sample.get("case", ""))
    if case:
        return case
    path = sample.get("target_img", "") or sample.get("target_mask", "")
    match = re.search(r"(patient[_-]?\d+|case[_-]?\d+)", path, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def frame_name_from_sample(sample):
    return os.path.basename(sample.get("target_img") or sample.get("target_mask") or "")


def tensor_to_display_image(seq_tensor, target_idx):
    idx = max(0, min(int(target_idx), seq_tensor.shape[0] - 1))
    image = seq_tensor[idx, 0].detach().cpu().numpy().astype(np.float32)
    lo, hi = np.percentile(image, [1, 99])
    if hi <= lo:
        hi = image.max() if image.max() > lo else lo + 1.0
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def count_components(mask):
    mask = np.asarray(mask).astype(np.uint8)
    if mask.max() == 0:
        return 0
    return int(cv2.connectedComponents(mask, connectivity=8)[0] - 1)


def safe_ratio(num, den):
    return float(num / den) if den > 0 else float("nan")


def dice_from_masks(pred, gt, eps=1e-7):
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return float((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps))


def confusion_counts(pred, gt):
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    tn = float(np.logical_and(~pred, ~gt).sum())
    return tp, fp, tn, fn


def metrics_from_counts(tp, fp, tn, fn, pred_fg, gt_fg, hd95_values, eps=1e-7):
    return {
        "Dice": (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps),
        "Precision": (tp + eps) / (tp + fp + eps),
        "Recall": (tp + eps) / (tp + fn + eps),
        "Specificity": (tn + eps) / (tn + fp + eps),
        "FPR": (fp + eps) / (fp + tn + eps),
        "FNR": (fn + eps) / (fn + tp + eps),
        "Pred/GT foreground ratio": safe_ratio(pred_fg, gt_fg),
        "HD95": float(np.nanmean(hd95_values)) if hd95_values else float("nan"),
    }


def build_dataset(args, cfg):
    return SequencePMTMDataset(
        root_dir=args.test_dir,
        window_size=int(cfg["data"].get("window_size", 5)),
        img_size=int(cfg["data"].get("img_size", 512)),
        is_train=False,
    )


def collect_sample_info(dataset, target_idx, max_samples=None):
    limit = len(dataset) if max_samples is None else min(len(dataset), int(max_samples))
    records = []
    images = []
    gts = []
    for idx in range(limit):
        seq_tensor, mask_tensor = dataset[idx]
        gt = (mask_tensor[0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        sample = dataset.samples[idx]
        gt_area = int(gt.sum())
        total = int(gt.size)
        records.append({
            "sample_index": idx,
            "frame_name": frame_name_from_sample(sample),
            "patient_id": patient_id_from_sample(sample),
            "target_img": sample.get("target_img", ""),
            "target_mask": sample.get("target_mask", ""),
            "gt_area_ratio": safe_ratio(gt_area, total),
            "gt_foreground_pixels": gt_area,
            "image_pixels": total,
            "connected_components": count_components(gt),
            "phase": "",
        })
        images.append(tensor_to_display_image(seq_tensor, target_idx))
        gts.append(gt)
    assign_phases(records)
    return records, images, gts


def assign_phases(records):
    broken = [r["gt_area_ratio"] for r in records if r["gt_foreground_pixels"] > 0]
    if not broken:
        for r in records:
            r["phase"] = "Non-broken"
        return
    q33, q66 = np.quantile(np.asarray(broken, dtype=np.float64), [1.0 / 3.0, 2.0 / 3.0])
    for r in records:
        if r["gt_foreground_pixels"] <= 0:
            r["phase"] = "Non-broken"
        elif r["gt_area_ratio"] <= q33:
            r["phase"] = "Early/Small"
        elif r["gt_area_ratio"] <= q66:
            r["phase"] = "Middle/Medium"
        else:
            r["phase"] = "Late/Large"
    for r in records:
        r["q33_gt_area_ratio"] = float(q33)
        r["q66_gt_area_ratio"] = float(q66)


def select_model_input(image_seq, model_type, target_idx):
    if model_type == "static":
        return image_seq[:, target_idx]
    if model_type == "temporal":
        return image_seq
    raise ValueError(f"Unsupported model type: {model_type}")


def load_model(model_name, cfg, checkpoint_dir, device):
    ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    model = build_registered_model(model_name, cfg).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    if state and next(iter(state)).startswith("module."):
        state = {k[7:]: v for k, v in state.items()}
    load_info = model.load_state_dict(state, strict=False)
    missing = len(getattr(load_info, "missing_keys", []))
    unexpected = len(getattr(load_info, "unexpected_keys", []))
    print(f"[CKPT] {model_name}: {ckpt_path} | missing={missing} unexpected={unexpected}")
    model.eval()
    return model


def prediction_path(output_dir, model_name, sample_index):
    return os.path.join(output_dir, "predictions", model_name, f"pred_{sample_index:04d}.png")


def prediction_root(args):
    return args.prediction_dir or os.path.join(args.output_dir, "predictions")


def prediction_path_for_args(args, model_name, sample_index):
    return os.path.join(prediction_root(args), model_name, f"pred_{sample_index:04d}.png")


def run_inference_for_model(args, cfg, dataset, model_name, records, target_idx, device):
    pred_dir = os.path.join(prediction_root(args), model_name)
    os.makedirs(pred_dir, exist_ok=True)
    expected = [prediction_path_for_args(args, model_name, r["sample_index"]) for r in records]
    if args.skip_inference:
        missing = [path for path in expected if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(f"--skip_inference was set but predictions are missing, e.g. {missing[0]}")
        print(f"[Skip] Using existing predictions for {model_name}: {pred_dir}")
        return
    if not args.force and all(os.path.exists(path) for path in expected):
        print(f"[Skip] Reusing saved predictions for {model_name}: {pred_dir}")
        return

    model = load_model(model_name, cfg, args.checkpoint_dir, device)
    model_type = cfg.get("models", {}).get(model_name, {}).get("type", "temporal")
    threshold = float(cfg["eval"].get("threshold", 0.5))
    use_amp = bool(cfg.get("training", {}).get("amp", True)) and device.type == "cuda"
    batch_size = max(1, int(args.batch_size))

    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch_records = records[start:start + batch_size]
            seqs = [dataset[r["sample_index"]][0] for r in batch_records]
            image_seq = torch.stack(seqs, dim=0).to(device)
            x = select_model_input(image_seq, model_type, target_idx)
            autocast_ctx = torch.amp.autocast(device_type="cuda") if use_amp else nullcontext()
            with autocast_ctx:
                logits = get_logits(model(x))
            if logits.shape[-2:] != image_seq.shape[-2:]:
                logits = F.interpolate(logits, size=image_seq.shape[-2:], mode="bilinear", align_corners=False)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            for item, record in zip(probs, batch_records):
                pred = (item.squeeze() > threshold).astype(np.uint8) * 255
                cv2.imwrite(prediction_path_for_args(args, model_name, record["sample_index"]), pred)
            print(f"[Infer] {model_name}: {min(start + batch_size, len(records))}/{len(records)}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_prediction(args, model_name, sample_index):
    path = prediction_path_for_args(args, model_name, sample_index)
    pred = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if pred is None:
        raise FileNotFoundError(path)
    return (pred > 127).astype(np.uint8)


def evaluate_phase_metrics(args, models, records, gts):
    phase_order = ["Early/Small", "Middle/Medium", "Late/Large", "Non-broken"]
    rows = []
    sample_model_metrics = {}
    for model_name in models:
        for phase in phase_order:
            phase_records = [r for r in records if r["phase"] == phase]
            if not phase_records:
                continue
            tp = fp = tn = fn = 0.0
            pred_fg = gt_fg = 0.0
            hd95_values = []
            gt_area_ratios = []
            for r in phase_records:
                idx = r["sample_index"]
                gt = gts[idx]
                pred = load_prediction(args, model_name, idx)
                item_tp, item_fp, item_tn, item_fn = confusion_counts(pred, gt)
                tp += item_tp
                fp += item_fp
                tn += item_tn
                fn += item_fn
                pred_fg += float(pred.astype(bool).sum())
                gt_fg += float(gt.astype(bool).sum())
                hd95_values.append(hd95(pred, gt))
                gt_area_ratios.append(r["gt_area_ratio"])
                sample_model_metrics[(model_name, idx)] = {
                    "dice": dice_from_masks(pred, gt),
                    "hd95": hd95_values[-1],
                }
            metrics = metrics_from_counts(tp, fp, tn, fn, pred_fg, gt_fg, hd95_values)
            row = {
                "model": model_name,
                "display_name": DISPLAY_NAMES.get(model_name, model_name),
                "phase": phase,
                "sample_count": len(phase_records),
                "mean_gt_area_ratio": float(np.mean(gt_area_ratios)),
                "metric_average_mode": METRIC_AVERAGE_MODE,
            }
            row.update(metrics)
            rows.append(row)
    return rows, sample_model_metrics


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_sample_info(output_dir, records):
    fields = [
        "sample_index", "frame_name", "patient_id", "target_img", "target_mask",
        "gt_area_ratio", "gt_foreground_pixels", "image_pixels", "connected_components",
        "phase", "q33_gt_area_ratio", "q66_gt_area_ratio",
    ]
    write_csv(os.path.join(output_dir, "phasewise_sample_info.csv"), records, fields)


def write_metrics_long(output_dir, rows):
    fields = [
        "model", "display_name", "phase", "sample_count", "mean_gt_area_ratio",
        "Dice", "Precision", "Recall", "Specificity", "FPR", "FNR",
        "Pred/GT foreground ratio", "HD95", "metric_average_mode",
    ]
    write_csv(os.path.join(output_dir, "phasewise_metrics_long.csv"), rows, fields)


def write_metrics_wide(output_dir, rows):
    metrics = ["Dice", "Precision", "Recall", "Specificity", "FPR", "FNR", "Pred/GT foreground ratio", "HD95"]
    by_model = defaultdict(dict)
    names = {}
    for row in rows:
        model = row["model"]
        names[model] = row["display_name"]
        by_model[model][("sample_count", row["phase"])] = row["sample_count"]
        for metric in metrics:
            by_model[model][(metric, row["phase"])] = row[metric]

    phases = ["Early/Small", "Middle/Medium", "Late/Large", "Non-broken"]
    fields = ["model", "display_name"]
    for phase in phases:
        fields.append(f"{phase}_sample_count")
        for metric in metrics:
            fields.append(f"{phase}_{metric}")

    wide_rows = []
    for model, values in by_model.items():
        row = {"model": model, "display_name": names.get(model, model)}
        for phase in phases:
            row[f"{phase}_sample_count"] = values.get(("sample_count", phase), "")
            for metric in metrics:
                row[f"{phase}_{metric}"] = values.get((metric, phase), "")
        wide_rows.append(row)
    write_csv(os.path.join(output_dir, "phasewise_metrics_wide.csv"), wide_rows, fields)


def fmt(value):
    if value == "":
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        value = float(value)
    except Exception:
        return str(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def markdown_table(rows, columns):
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_markdown(output_dir, rows):
    columns = [
        "display_name", "phase", "sample_count", "mean_gt_area_ratio", "Dice",
        "Precision", "Recall", "Specificity", "FPR", "FNR",
        "Pred/GT foreground ratio", "HD95",
    ]
    path = os.path.join(output_dir, "phasewise_results.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Phase-wise Evaluation Results\n\n")
        f.write(f"- metric_average_mode: `{METRIC_AVERAGE_MODE}`\n")
        f.write("- Dice/Precision/Recall/Specificity/FPR/FNR/Pred-GT foreground ratio use accumulated TP/FP/FN/TN over each subset.\n")
        f.write("- HD95 is computed per image and then averaged within each subset.\n\n")
        f.write(markdown_table(rows, columns))
        f.write("\n")


def plot_metric_bars(output_dir, rows, metric, filename, dpi):
    phases = ["Early/Small", "Middle/Medium", "Late/Large"]
    models = []
    for row in rows:
        if row["phase"] in phases and row["model"] not in models:
            models.append(row["model"])
    if not models:
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.text(0.5, 0.5, f"No {metric} data for broken phases", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), dpi=dpi)
        plt.close(fig)
        return
    values = {(row["model"], row["phase"]): row.get(metric, np.nan) for row in rows}

    x = np.arange(len(phases))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.2 * len(models) + 3.5), 4.2))
    for i, model in enumerate(models):
        ys = [values.get((model, phase), np.nan) for phase in phases]
        ax.bar(x - 0.4 + width / 2 + i * width, ys, width, label=DISPLAY_NAMES.get(model, model))
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=dpi)
    plt.close(fig)


def choose_representative_samples(records, sample_model_metrics):
    by_phase = defaultdict(list)
    for r in records:
        if r["phase"] in {"Early/Small", "Middle/Medium", "Late/Large"}:
            by_phase[r["phase"]].append(r)

    selected = {}
    for phase, items in by_phase.items():
        if phase == "Early/Small":
            ranked = [
                (
                    r["gt_area_ratio"],
                    -(
                        sample_model_metrics.get(("siamese_stpnet", r["sample_index"]), {}).get("dice", 0.0)
                        - sample_model_metrics.get(("unet3plus", r["sample_index"]), {}).get("dice", 0.0)
                    ),
                    r,
                )
                for r in items
                if sample_model_metrics.get(("siamese_stpnet", r["sample_index"]), {}).get("dice", 0.0)
                > sample_model_metrics.get(("unet3plus", r["sample_index"]), {}).get("dice", 0.0)
            ]
            if len(ranked) < 3:
                ranked = [(r["gt_area_ratio"], 0.0, r) for r in items]
            selected[phase] = [item[-1] for item in sorted(ranked)[:3]]
        elif phase == "Middle/Medium":
            diffs = []
            for r in items:
                diff = abs(
                    sample_model_metrics.get(("siamese_stpnet", r["sample_index"]), {}).get("dice", 0.0)
                    - sample_model_metrics.get(("unet3plus", r["sample_index"]), {}).get("dice", 0.0)
                )
                diffs.append((diff, r))
            diffs = sorted(diffs, key=lambda x: x[0])
            if diffs:
                mid = len(diffs) // 2
                picked = diffs[max(0, mid - 1):mid + 2]
                selected[phase] = [r for _, r in picked][:3]
            else:
                selected[phase] = []
        else:
            selected[phase] = sorted(items, key=lambda r: r["gt_area_ratio"], reverse=True)[:3]
    return selected


def qualitative_columns(models):
    preferred = ["unet", "unet3plus", "2_5d_unet", "siamese_biconvlstm", "siamese_stpnet"]
    return [m for m in preferred if m in models]


def save_qualitative(args, phase_name, samples, models, images, gts, dpi):
    if not samples:
        return
    model_cols = qualitative_columns(models)
    columns = ["Input", "GT"] + model_cols
    fig, axes = plt.subplots(len(samples), len(columns), figsize=(1.45 * len(columns), 1.45 * len(samples) + 0.35), squeeze=False)
    for row, record in enumerate(samples):
        idx = record["sample_index"]
        image = images[idx]
        gt = gts[idx]
        for col, column in enumerate(columns):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if column == "Input":
                ax.imshow(np.repeat(image[..., None], 3, axis=-1))
                ax.text(0.02, 0.96, record["frame_name"], transform=ax.transAxes, ha="left", va="top", fontsize=6, color="white", bbox=dict(facecolor="#202020", edgecolor="none", alpha=0.8, pad=2))
            elif column == "GT":
                ax.imshow(overlay_mask(image, gt, None, alpha=0.48))
            else:
                pred = load_prediction(args, column, idx)
                ax.imshow(overlay_mask(image, pred, gt, alpha=0.42))
            if row == 0:
                title = column if column in {"Input", "GT"} else DISPLAY_NAMES.get(column, column)
                ax.set_title(title, fontsize=7, pad=4)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.02, wspace=0.02, hspace=0.03)
    filename = {
        "Early/Small": "qualitative_early.png",
        "Middle/Medium": "qualitative_middle.png",
        "Late/Large": "qualitative_late.png",
    }[phase_name]
    fig.savefig(os.path.join(args.output_dir, filename), dpi=dpi)
    plt.close(fig)


def write_hard_subset(args, records, rows, sample_model_metrics):
    broken = [r for r in records if r["gt_foreground_pixels"] > 0]
    hard_fields = [
        "sample_index", "frame_name", "patient_id", "gt_area_ratio", "phase",
        "connected_components", "unet3plus_dice", "unet3plus_overall_mean_dice", "hard_reasons",
        "target_img", "target_mask",
    ]
    if not broken:
        write_csv(os.path.join(args.output_dir, "hard_subset_list.csv"), [], hard_fields)
        with open(os.path.join(args.output_dir, "hard_subset_metrics.md"), "w", encoding="utf-8") as f:
            f.write("# Hard Subset Metrics\n\n")
            f.write("- hard sample count: `0`\n")
            f.write(f"- metric_average_mode: `{METRIC_AVERAGE_MODE}`\n")
        return
    q30 = float(np.quantile([r["gt_area_ratio"] for r in broken], 0.30))
    unet3plus_values = [
        sample_model_metrics.get(("unet3plus", r["sample_index"]), {}).get("dice", np.nan)
        for r in broken
    ]
    unet3plus_mean = float(np.nanmean(unet3plus_values)) if unet3plus_values else float("nan")
    hard_records = []
    for r in broken:
        dice_value = sample_model_metrics.get(("unet3plus", r["sample_index"]), {}).get("dice", np.nan)
        reasons = []
        if r["gt_area_ratio"] <= q30:
            reasons.append("small_gt_area_bottom_30pct")
        if r["connected_components"] >= 2:
            reasons.append("connected_components_ge_2")
        if not math.isnan(dice_value) and dice_value <= unet3plus_mean - float(args.hard_dice_drop):
            reasons.append("unet3plus_dice_below_mean")
        if reasons:
            item = dict(r)
            item["unet3plus_dice"] = dice_value
            item["unet3plus_overall_mean_dice"] = unet3plus_mean
            item["hard_reasons"] = ";".join(reasons)
            hard_records.append(item)

    write_csv(os.path.join(args.output_dir, "hard_subset_list.csv"), hard_records, hard_fields)

    hard_indices = {r["sample_index"] for r in hard_records}
    hard_rows = []
    models = sorted({row["model"] for row in rows})
    for model in models:
        tp = fp = tn = fn = pred_fg = gt_fg = 0.0
        hd95_values = []
        for r in hard_records:
            idx = r["sample_index"]
            gt = load_gt_cache[idx]
            pred = load_prediction(args, model, idx)
            item_tp, item_fp, item_tn, item_fn = confusion_counts(pred, gt)
            tp += item_tp
            fp += item_fp
            tn += item_tn
            fn += item_fn
            pred_fg += float(pred.astype(bool).sum())
            gt_fg += float(gt.astype(bool).sum())
            hd95_values.append(hd95(pred, gt))
        if hard_indices:
            metrics = metrics_from_counts(tp, fp, tn, fn, pred_fg, gt_fg, hd95_values)
            row = {
                "display_name": DISPLAY_NAMES.get(model, model),
                "sample_count": len(hard_records),
                "metric_average_mode": METRIC_AVERAGE_MODE,
            }
            row.update(metrics)
            hard_rows.append(row)

    columns = [
        "display_name", "sample_count", "Dice", "Precision", "Recall", "Specificity",
        "FPR", "FNR", "Pred/GT foreground ratio", "HD95", "metric_average_mode",
    ]
    with open(os.path.join(args.output_dir, "hard_subset_metrics.md"), "w", encoding="utf-8") as f:
        f.write("# Hard Subset Metrics\n\n")
        f.write(f"- hard sample count: `{len(hard_records)}`\n")
        f.write(f"- bottom-30% gt_area_ratio threshold: `{q30:.6f}`\n")
        f.write(f"- UNet 3+ overall mean Dice: `{unet3plus_mean:.4f}`\n")
        f.write(f"- UNet 3+ hard Dice condition: `dice <= mean - {float(args.hard_dice_drop):.4f}`\n")
        f.write(f"- metric_average_mode: `{METRIC_AVERAGE_MODE}`\n\n")
        f.write(markdown_table(hard_rows, columns))
        f.write("\n")


load_gt_cache = []


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    models = [normalize_model_name(name) for name in args.models]
    cfg = prepare_config(args)
    device = resolve_device(args.device)
    target_idx = int(cfg["data"].get("target_idx", int(cfg["data"].get("window_size", 5)) // 2))

    dataset = build_dataset(args, cfg)
    records, images, gts = collect_sample_info(dataset, target_idx, max_samples=args.max_samples)
    global load_gt_cache
    load_gt_cache = gts
    write_sample_info(args.output_dir, records)

    with open(os.path.join(args.output_dir, "phasewise_run_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "test_dir": args.test_dir,
            "checkpoint_dir": args.checkpoint_dir,
            "models": models,
            "threshold": float(cfg["eval"].get("threshold", 0.5)),
            "metric_average_mode": METRIC_AVERAGE_MODE,
        }, f, indent=2, ensure_ascii=False)

    for model_name in models:
        run_inference_for_model(args, cfg, dataset, model_name, records, target_idx, device)

    metric_rows, sample_model_metrics = evaluate_phase_metrics(args, models, records, gts)
    write_metrics_long(args.output_dir, metric_rows)
    write_metrics_wide(args.output_dir, metric_rows)
    write_markdown(args.output_dir, metric_rows)
    plot_metric_bars(args.output_dir, metric_rows, "Dice", "phasewise_dice_bar.png", args.dpi)
    plot_metric_bars(args.output_dir, metric_rows, "HD95", "phasewise_hd95_bar.png", args.dpi)

    selected = choose_representative_samples(records, sample_model_metrics)
    for phase_name, samples in selected.items():
        save_qualitative(args, phase_name, samples, models, images, gts, args.dpi)

    write_hard_subset(args, records, metric_rows, sample_model_metrics)
    print(f"Wrote phase-wise outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

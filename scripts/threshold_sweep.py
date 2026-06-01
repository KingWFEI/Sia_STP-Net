import argparse
import csv
import importlib
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train import (  # noqa: E402
    SequencePMTMDataset,
    collect_output_diagnostics,
    compute_all_metrics,
    compute_batch_hd95,
    nan_value,
    safe_collate,
)


DEFAULT_THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run validation threshold sweep for a trained segmentation checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--module", type=str, default="net.sia_prompt_net_bdf_v4")
    parser.add_argument("--checkpoint", type=str, default=r"runs\bdf_all_versions\v4\best_dice.pth")
    parser.add_argument("--val_dir", type=str, default=r"D:\datasets\pmtm\TBUT_Seg_Data_v1\val")
    parser.add_argument("--output_dir", type=str, default=r"runs\bdf_all_versions\v4\threshold_sweep")
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--use_polar", type=str2bool, default=False)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--compute_hd95", action="store_true", default=False)
    return parser.parse_args()


def avg_optional(optional_totals, optional_counts):
    return {
        key: optional_totals[key] / optional_counts[key] if optional_counts[key] > 0 else nan_value()
        for key in optional_totals
    }


def load_model(args, device):
    module = importlib.import_module(args.module)
    model = module.build_model(
        window_size=args.window_size,
        image_size=(args.img_size, args.img_size),
        num_classes=1,
        input_channels=1,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Load] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[Load] Unexpected keys: {len(unexpected)}")
    return model


def sweep(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.amp

    dataset = SequencePMTMDataset(
        root_dir=args.val_dir,
        window_size=args.window_size,
        img_size=args.img_size,
        use_polar=args.use_polar,
        is_train=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=safe_collate,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = load_model(args, device)
    model.eval()
    if hasattr(model, "return_diagnostics"):
        model.return_diagnostics = True

    metric_keys = [
        "dice", "iou", "recall", "precision", "specificity",
        "fpr", "fnr", "pred_fg_ratio", "gt_fg_ratio", "pred_gt_fg_ratio",
        "TP", "FP", "FN", "TN",
    ]
    totals = {thr: {key: 0.0 for key in metric_keys} for thr in args.thresholds}
    hd95_values = {thr: [] for thr in args.thresholds}
    optional_keys = [
        "gate_mean", "gate_std", "gate_min", "gate_max",
        "prompt_norm", "target_feat_norm", "merged_state_norm", "prompt_target_ratio",
    ]
    optional_totals = {key: 0.0 for key in optional_keys}
    optional_counts = {key: 0 for key in optional_keys}
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Threshold sweep", ncols=110):
            if batch is None:
                continue
            seq_imgs, masks = batch
            seq_imgs = seq_imgs.to(device)
            masks = masks.to(device)

            with autocast("cuda", enabled=use_amp):
                outputs = model(seq_imgs)

            logits = outputs["seg"] if isinstance(outputs, dict) else outputs
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            for threshold in args.thresholds:
                metrics = compute_all_metrics(logits, masks, threshold=threshold)
                for key in metric_keys:
                    totals[threshold][key] += metrics[key]
                if args.compute_hd95:
                    hd95 = compute_batch_hd95(logits, masks, threshold=threshold)
                    if not math.isnan(hd95):
                        hd95_values[threshold].append(hd95)

            opt_stats = collect_output_diagnostics(outputs)
            for key, value in opt_stats.items():
                if key in optional_totals and not math.isnan(value):
                    optional_totals[key] += value
                    optional_counts[key] += 1
            n_batches += 1

    gate_stats = avg_optional(optional_totals, optional_counts)
    rows = []
    for threshold in args.thresholds:
        row = {"threshold": threshold}
        for key in metric_keys:
            row[key] = totals[threshold][key] / max(n_batches, 1)
        row["hd95"] = float(sum(hd95_values[threshold]) / len(hd95_values[threshold])) if hd95_values[threshold] else nan_value()
        row.update(gate_stats)
        rows.append(row)

    rows.sort(key=lambda row: row["threshold"])
    write_outputs(args, rows)
    return rows


def fmt(value, digits=4):
    if value is None:
        return "N/A"
    try:
        value = float(value)
        if math.isnan(value):
            return "N/A"
        return f"{value:.{digits}f}"
    except Exception:
        return "N/A"


def write_outputs(args, rows):
    csv_path = os.path.join(args.output_dir, "threshold_sweep.csv")
    json_path = os.path.join(args.output_dir, "threshold_sweep.json")
    md_path = os.path.join(args.output_dir, "threshold_sweep.md")

    fieldnames = list(rows[0].keys()) if rows else ["threshold"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda row: row["dice"]) if rows else {}
    payload = {
        "checkpoint": args.checkpoint,
        "module": args.module,
        "val_dir": args.val_dir,
        "thresholds": args.thresholds,
        "best_by_dice": best,
        "rows": rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Threshold Sweep\n\n")
        f.write(f"- checkpoint: `{args.checkpoint}`\n")
        f.write(f"- module: `{args.module}`\n")
        if best:
            f.write(
                f"- best by Dice: threshold={fmt(best['threshold'], 2)}, "
                f"Dice={fmt(best['dice'])}, Precision={fmt(best['precision'])}, Recall={fmt(best['recall'])}\n"
            )
        f.write("\n")
        f.write("| threshold | dice | iou | precision | recall | fpr | fnr | pred/gt | hd95 |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            f.write(
                f"| {fmt(row['threshold'], 2)} | {fmt(row['dice'])} | {fmt(row['iou'])} | "
                f"{fmt(row['precision'])} | {fmt(row['recall'])} | {fmt(row['fpr'])} | "
                f"{fmt(row['fnr'])} | {fmt(row['pred_gt_fg_ratio'])} | {fmt(row['hd95'])} |\n"
            )

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    if best:
        print(
            "Best by Dice: "
            f"threshold={best['threshold']:.2f}, dice={best['dice']:.4f}, "
            f"precision={best['precision']:.4f}, recall={best['recall']:.4f}"
        )


def main():
    args = parse_args()
    sweep(args)


if __name__ == "__main__":
    main()

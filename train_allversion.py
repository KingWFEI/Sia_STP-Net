import argparse
import csv
import gc
import importlib
import json
import os
import traceback
from datetime import datetime
from types import SimpleNamespace

import torch

import train


BDF_VARIANTS = {
    "v1": {
        "module": "net.sia_prompt_net_bdf",
        "display": "BDF-v1",
        "desc": "baseline: e_hat_t = r_t * e_t",
    },
    "v1_1": {
        "module": "net.sia_prompt_net_bdf_v1_1",
        "display": "BDF-v1-1",
        "desc": "v1 + shallow reliability-gated skip",
    },
    "v2": {
        "module": "net.sia_prompt_net_bdf_v2",
        "display": "BDF-v2",
        "desc": "recall-oriented: e_hat_t = (0.5 + 0.5*r_t) * e_t",
    },
    "v3": {
        "module": "net.sia_prompt_net_bdf_v3",
        "display": "BDF-v3",
        "desc": "target_project for feature distribution alignment",
    },
    "v4": {
        "module": "net.sia_prompt_net_bdf_v4",
        "display": "BDF-v4",
        "desc": "v2 + learnable gamma residual",
    },
    "v5": {
        "module": "net.sia_prompt_net_bdf_v5",
        "display": "BDF-v5",
        "desc": "v4 + target_project evidence for target frame",
    },
}


SUMMARY_FIELDS = [
    "version",
    "display",
    "status",
    "save_dir",
    "best_epoch_by_val_dice",
    "best_val_dice",
    "val_iou",
    "val_precision",
    "val_recall",
    "val_specificity",
    "val_fpr",
    "val_fnr",
    "val_hd95",
    "pred_fg_ratio",
    "gt_fg_ratio",
    "pred_gt_fg_ratio",
    "params_m",
    "max_gpu_memory_mb",
    "epoch_time_sec",
    "defect_report",
    "error",
]


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Siamese ST-Prompt-Net BDF variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--train_dir", type=str, default=r"D:\datasets\pmtm\TBUT_Seg_Data_v1\train")
    parser.add_argument("--val_dir", type=str, default=r"D:\datasets\pmtm\TBUT_Seg_Data_v1\val")
    parser.add_argument("--base_save_dir", type=str, default="./runs/bdf_all_versions")
    parser.add_argument("--versions", nargs="+", default=["v1", "v2", "v3", "v4", "v5"], choices=list(BDF_VARIANTS))
    parser.add_argument("--version", dest="versions", nargs="+", choices=list(BDF_VARIANTS), default=argparse.SUPPRESS)

    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--use_polar", type=str2bool, default=False)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pos_weight", type=float, default=2.0)
    parser.add_argument("--early_stopping", type=int, default=30)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--compute_hd95", action="store_true", default=False)
    parser.add_argument("--hd95_every", type=int, default=1)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--force", action="store_true", default=False, help="Re-run variants with existing best_dice.pth.")
    return parser.parse_args()


def make_train_args(args, version):
    save_dir = os.path.join(args.base_save_dir, version)
    return SimpleNamespace(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        save_dir=save_dir,
        window_size=args.window_size,
        img_size=args.img_size,
        use_polar=args.use_polar,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        pretrained=args.pretrained,
        resume=args.resume,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        pos_weight=args.pos_weight,
        early_stopping=args.early_stopping,
        amp=args.amp,
        weight_decay=args.weight_decay,
        net_name=BDF_VARIANTS[version]["display"],
        enable_diagnostics=True,
        compute_hd95=args.compute_hd95,
        hd95_every=args.hd95_every,
        diagnostics_dir=None,
    )


def run_variant(args, version):
    info = BDF_VARIANTS[version]
    save_dir = os.path.join(args.base_save_dir, version)
    best_path = os.path.join(save_dir, "best_dice.pth")
    if os.path.exists(best_path) and not args.force:
        print(f"[Skip] {info['display']} already has best_dice.pth. Use --force to re-run.")
        return load_summary_row(version, save_dir, status="skipped")

    module = importlib.import_module(info["module"])
    train.build_model = module.build_model

    print("\n" + "=" * 80)
    print(f"Running {info['display']}: {info['desc']}")
    print(f"Module  : {info['module']}")
    print(f"Save dir: {save_dir}")
    print("=" * 80)

    train_args = make_train_args(args, version)
    os.makedirs(save_dir, exist_ok=True)
    train.main(train_args)
    return load_summary_row(version, save_dir, status="completed")


def load_summary_row(version, save_dir, status, error=""):
    info = BDF_VARIANTS[version]
    row = {field: "" for field in SUMMARY_FIELDS}
    row.update({
        "version": version,
        "display": info["display"],
        "status": status,
        "save_dir": save_dir,
        "defect_report": os.path.join(save_dir, "diagnostics", "defect_report.md"),
        "error": error,
    })

    summary_path = os.path.join(save_dir, "diagnostics", "best_epoch_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        for key, value in summary.items():
            if key in row:
                row[key] = value
    return row


def write_summary(rows, base_save_dir):
    os.makedirs(base_save_dir, exist_ok=True)
    csv_path = os.path.join(base_save_dir, "bdf_versions_summary.csv")
    json_path = os.path.join(base_save_dir, "bdf_versions_summary.json")
    md_path = os.path.join(base_save_dir, "bdf_versions_summary.md")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    md_cols = [
        "version", "display", "status", "best_epoch_by_val_dice", "best_val_dice",
        "val_precision", "val_recall", "val_fpr", "val_fnr", "pred_gt_fg_ratio",
        "val_hd95", "max_gpu_memory_mb",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# BDF Versions Summary\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| " + " | ".join(md_cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(md_cols)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(format_cell(row.get(col, "")) for col in md_cols) + " |\n")
        f.write("\n## Defect Reports\n\n")
        for row in rows:
            f.write(f"- {row['display']}: `{row['defect_report']}`\n")

    print(f"\nSummary saved to: {md_path}")


def format_cell(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main():
    args = parse_args()
    os.makedirs(args.base_save_dir, exist_ok=True)
    rows = []
    error_log = os.path.join(args.base_save_dir, "errors.log")

    for version in args.versions:
        try:
            rows.append(run_variant(args, version))
        except Exception as exc:
            error = traceback.format_exc()
            print(f"[ERROR] {BDF_VARIANTS[version]['display']} failed. Continuing. See {error_log}")
            with open(error_log, "a", encoding="utf-8") as f:
                f.write(f"\n[{version}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(error)
                f.write("\n")
            rows.append(load_summary_row(version, os.path.join(args.base_save_dir, version), status="failed", error=str(exc)))
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_summary(rows, args.base_save_dir)


if __name__ == "__main__":
    main()

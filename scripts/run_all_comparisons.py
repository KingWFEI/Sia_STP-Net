import argparse
import copy
import os
import sys

import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from comparison.engine import (
    evaluate_one_model,
    log_exception,
    make_output_dirs,
    set_seed,
    train_one_model,
)
from comparison.export import export_results
from comparison.model_registry import MODEL_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser(description="Run all segmentation comparison experiments.")
    parser.add_argument("--config", type=str, required=True, help="Path to compare_models.yaml")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing best checkpoints.")
    parser.add_argument("--resume", action="store_true", help="Resume from *_last.pth when available.")
    parser.add_argument("--models", nargs="+", default=None, help="Subset of model keys to run.")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg.get("data", {}).get("test_dir"):
        cfg["data"]["test_dir"] = cfg["data"]["val_dir"]
    cfg["data"]["target_idx"] = int(cfg["data"].get("target_idx", int(cfg["data"]["window_size"]) // 2))
    return cfg


def selected_models(cfg, requested):
    model_names = list(cfg["models"].keys()) if requested is None else requested
    unknown = [name for name in model_names if name not in MODEL_REGISTRY]
    if unknown:
        raise KeyError(f"Unknown model(s): {unknown}. Available: {sorted(MODEL_REGISTRY)}")
    missing_cfg = [name for name in model_names if name not in cfg["models"]]
    if missing_cfg:
        raise KeyError(f"Model(s) missing from config models section: {missing_cfg}")
    return model_names


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    dirs = make_output_dirs(cfg.get("output_dir", "run/comparison"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    rows = []
    error_log = os.path.join(dirs["logs"], "errors.log")
    models = selected_models(cfg, args.models)

    for model_name in models:
        print("\n" + "=" * 80)
        print(f"Running model: {model_name}")
        print("=" * 80)
        best_ckpt = os.path.join(dirs["checkpoints"], f"{model_name}_best.pth")
        try:
            if args.eval_only:
                row = evaluate_one_model(model_name, cfg, dirs, device)
            elif cfg["training"].get("skip_completed", True) and os.path.exists(best_ckpt) and not args.resume:
                print(f"Skip training for completed model: {model_name}; evaluating best checkpoint.")
                row = evaluate_one_model(model_name, cfg, dirs, device)
            else:
                row = run_with_oom_fallback(model_name, cfg, dirs, device, args.resume)
            rows.append(row)
            export_results(rows, dirs["metrics"], os.path.join(dirs["root"], "summary.md"))
        except Exception:
            print(f"[ERROR] {model_name} failed. Traceback saved to {error_log}. Continuing.")
            log_exception(error_log, model_name)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    export_results(rows, dirs["metrics"], os.path.join(dirs["root"], "summary.md"))
    print(f"\nFinished. Summary: {os.path.join(dirs['root'], 'summary.md')}")


def run_with_oom_fallback(model_name, cfg, dirs, device, resume):
    try:
        return train_one_model(model_name, cfg, dirs, device, resume=resume)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" not in msg or int(cfg["training"]["batch_size"]) <= 1:
            raise
        print(f"[OOM] {model_name}: retrying with batch_size=1, keeping other hyperparameters unchanged.")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        cfg_retry = copy.deepcopy(cfg)
        cfg_retry["training"]["batch_size"] = 1
        return train_one_model(model_name, cfg_retry, dirs, device, resume=resume, batch_size=1)


if __name__ == "__main__":
    main()


import argparse
import json
import math
import os
import sys
import textwrap
from collections import OrderedDict
from contextlib import nullcontext

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from comparison.metrics import get_logits  # noqa: E402
from comparison.model_registry import build_registered_model  # noqa: E402
from utils.utils import SequencePMTMDataset  # noqa: E402
from utils.utils import get_cornea_circle_from_json  # noqa: E402


DEFAULT_MODEL_ORDER = [
    "unet",
    "unetpp",
    "deeplabv3plus",
    "unet3plus",
    "attention_unet",
    "transunet",
    "swinunet",
    "2_5d_unet",
    "siamese_encoder_decoder",
    "siamese_biconvlstm",
    "siamese_stpnet",
]

DEFAULT_DISPLAY_NAMES = {
    "unet": "U-Net",
    "unetpp": "U-Net++",
    "deeplabv3plus": "DeepLabV3+",
    "unet3plus": "UNet 3+",
    "attention_unet": "Att U-Net",
    "transunet": "TransUNet",
    "swinunet": "Swin-Unet",
    "2_5d_unet": "2.5D U-Net",
    "siamese_encoder_decoder": "Sia ED",
    "siamese_biconvlstm": "Sia BiLSTM",
    "siamese_stpnet": "STP-Net",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a paper-style segmentation comparison figure from saved prediction masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--comparison_dir", default=os.path.join("run", "comparison"))
    parser.add_argument("--patient_dir", default=None, help="Visualize a processed patient folder with images/json/overlays.")
    parser.add_argument("--checkpoint_dir", default=None, help="Directory containing *_best.pth checkpoints.")
    parser.add_argument(
        "--stpnet_ckpt",
        default=None,
        help="Optional checkpoint override for siamese_stpnet when visualizing processed patient folders.",
    )
    parser.add_argument("--patient_mode", choices=["inference", "labelme"], default="inference")
    parser.add_argument("--config", default=os.path.join("configs", "compare_models.yaml"))
    parser.add_argument("--val_dir", default=None, help="Override validation/test directory.")
    parser.add_argument("--output", default=os.path.join("run", "comparison", "figures", "segmentation_comparison.png"))
    parser.add_argument("--pdf", default=None, help="Optional PDF output path.")
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 3, 7, 11])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODEL_ORDER)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--target_idx", type=int, default=None)
    parser.add_argument("--max_cols", type=int, default=13, help="Reserved for future multi-page layouts.")
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--mask_alpha", type=float, default=0.42)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Prediction threshold in [0, 1]. Defaults to config eval.threshold, or 0.58 if absent. Saved-prediction mode also accepts 0-255 values.",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save_patient_predictions", action="store_true")
    parser.add_argument("--debug_patient_stats", action="store_true")
    parser.add_argument("--no_dice", action="store_true")
    parser.add_argument("--show_input_contour", action="store_true")
    parser.add_argument("--use_config_names", action="store_true", help="Use full display names from the YAML config.")
    return parser.parse_args()


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def load_yaml_config(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def resolve_threshold(args, cfg, fallback=0.58):
    if args.threshold is not None:
        return float(args.threshold)
    eval_cfg = cfg.get("eval", {}) if isinstance(cfg, dict) else {}
    if "threshold" in eval_cfg:
        return float(eval_cfg["threshold"])
    return float(fallback)


def resolve_settings(args):
    cfg = load_yaml_config(args.config)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("models", {})

    val_dir = args.val_dir or data_cfg.get("test_dir") or data_cfg.get("val_dir")
    if not val_dir:
        raise ValueError("Cannot resolve val_dir/test_dir. Pass --val_dir explicitly.")

    window_size = int(args.window_size or data_cfg.get("window_size", 5))
    img_size = int(args.img_size or data_cfg.get("img_size", 512))
    target_idx = int(args.target_idx if args.target_idx is not None else data_cfg.get("target_idx", window_size // 2))

    display_names = dict(DEFAULT_DISPLAY_NAMES)
    for name, item in model_cfg.items():
        if args.use_config_names and isinstance(item, dict) and item.get("display_name"):
            display_names[name] = item["display_name"]

    args.threshold = resolve_threshold(args, cfg)

    return val_dir, window_size, img_size, target_idx, display_names


def normalize_image(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        image = image.squeeze()
    lo, hi = np.percentile(image, [1, 99])
    if hi <= lo:
        hi = image.max() if image.max() > lo else lo + 1.0
    image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return image


def tensor_to_image(seq_tensor, target_idx):
    idx = max(0, min(int(target_idx), seq_tensor.shape[0] - 1))
    image = seq_tensor[idx, 0].detach().cpu().numpy()
    return normalize_image(image)


def tensor_to_mask(mask_tensor):
    mask = mask_tensor[0].detach().cpu().numpy()
    return (mask > 0.5).astype(np.uint8)


def read_pred_mask(path, threshold):
    pred = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if pred is None:
        return None
    threshold_value = float(threshold)
    if threshold_value <= 1.0:
        threshold_value *= 255.0
    return (pred > threshold_value).astype(np.uint8)


def resize_mask(mask, shape):
    if mask is None:
        return None
    if mask.shape == shape:
        return mask
    return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def dice_score(pred, target):
    if pred is None:
        return math.nan
    pred = pred.astype(bool)
    target = target.astype(bool)
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return 2.0 * np.logical_and(pred, target).sum() / denom


def mask_boundary(mask, width=2):
    mask = mask.astype(np.uint8)
    if mask.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    kernel = np.ones((3, 3), np.uint8)
    grad = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    if width > 1:
        grad = cv2.dilate(grad, np.ones((width, width), np.uint8), iterations=1)
    return grad > 0


FALSE_NEGATIVE_COLOR = (1.00, 0.86, 0.02)
FALSE_POSITIVE_COLOR = (0.05, 0.78, 0.18)
TRUE_SEGMENT_COLOR = (0.62, 0.18, 0.88)


def blend_region(rgb, region, color, alpha):
    if not region.any():
        return
    color = np.asarray(color, dtype=np.float32)
    rgb[region] = (1.0 - alpha) * rgb[region] + alpha * color


def overlay_mask(
    image,
    mask=None,
    gt=None,
    pred_color=TRUE_SEGMENT_COLOR,
    gt_color=TRUE_SEGMENT_COLOR,
    alpha=0.42,
):
    rgb = np.repeat(image[..., None], 3, axis=-1)
    rgb = 0.92 * rgb + 0.04

    if mask is not None and gt is not None:
        pred_region = mask.astype(bool)
        gt_region = gt.astype(bool)
        true_region = np.logical_and(pred_region, gt_region)
        false_negative = np.logical_and(~pred_region, gt_region)
        false_positive = np.logical_and(pred_region, ~gt_region)

        blend_region(rgb, true_region, TRUE_SEGMENT_COLOR, alpha)
        blend_region(rgb, false_negative, FALSE_NEGATIVE_COLOR, alpha)
        blend_region(rgb, false_positive, FALSE_POSITIVE_COLOR, alpha)

        boundary = mask_boundary(true_region.astype(np.uint8), width=2)
        boundary |= mask_boundary(false_negative.astype(np.uint8), width=2)
        boundary |= mask_boundary(false_positive.astype(np.uint8), width=2)
        rgb[boundary & true_region] = np.asarray(TRUE_SEGMENT_COLOR, dtype=np.float32)
        rgb[boundary & false_negative] = np.asarray(FALSE_NEGATIVE_COLOR, dtype=np.float32)
        rgb[boundary & false_positive] = np.asarray(FALSE_POSITIVE_COLOR, dtype=np.float32)
        return np.clip(rgb, 0.0, 1.0)

    if mask is not None and mask.any():
        color = np.asarray(pred_color, dtype=np.float32)
        mask_region = mask.astype(bool)
        blend_region(rgb, mask_region, color, alpha)
        rgb[mask_boundary(mask, width=2)] = color

    if gt is not None and gt.any():
        color = np.asarray(gt_color, dtype=np.float32)
        rgb[mask_boundary(gt, width=2)] = color

    return np.clip(rgb, 0.0, 1.0)


def read_rgb_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def image_to_display_rgb(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = normalize_image(gray)
    return np.repeat(gray[..., None], 3, axis=-1)


def load_labelme_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def draw_labelme_shapes(base_rgb, payload, alpha=0.34):
    canvas = base_rgb.copy()
    fill = canvas.copy()

    for shape in payload.get("shapes", []):
        label = shape.get("label")
        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if label == "broken" and len(points) >= 3:
            pts = np.round(points).astype(np.int32)
            cv2.fillPoly(fill, [pts], (1.0, 0.10, 0.08))
            cv2.polylines(canvas, [pts], True, (1.0, 0.05, 0.02), 2, cv2.LINE_AA)
        elif label == "cornea_roi" and len(points) >= 2:
            cx, cy = points[0]
            px, py = points[1]
            r = math.hypot(float(px - cx), float(py - cy))
            cv2.circle(canvas, (int(round(cx)), int(round(cy))), int(round(r)), (0.05, 0.86, 0.22), 2, cv2.LINE_AA)
            cv2.circle(canvas, (int(round(cx)), int(round(cy))), 3, (0.05, 0.86, 0.22), -1, cv2.LINE_AA)

    canvas = (1.0 - alpha) * canvas + alpha * fill
    return np.clip(canvas, 0.0, 1.0)


def count_broken(payload):
    return sum(
        1 for shape in payload.get("shapes", [])
        if shape.get("label") == "broken" and len(shape.get("points", [])) >= 3
    )


def list_image_paths(image_dir):
    return sorted(
        os.path.join(image_dir, name)
        for name in os.listdir(image_dir)
        if name.lower().endswith(IMAGE_EXTS)
    )


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(name)


def safe_circle(cx, cy, r, w, h, min_r=8.0):
    cx = float(np.clip(cx, 0, max(w - 1, 0)))
    cy = float(np.clip(cy, 0, max(h - 1, 0)))
    r = float(max(r, min_r))
    return cx, cy, r


def safe_rect(x1, y1, x2, y2, w, h):
    x1 = int(max(0, min(w - 1, round(x1))))
    y1 = int(max(0, min(h - 1, round(y1))))
    x2 = int(max(0, min(w, round(x2))))
    y2 = int(max(0, min(h, round(y2))))
    if x2 - x1 < 2:
        x1, x2 = 0, w
    if y2 - y1 < 2:
        y1, y2 = 0, h
    return x1, y1, x2, y2


def image_hw(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.shape[:2]


def estimate_patient_roi(image_paths, json_dir):
    h, w = image_hw(image_paths[0])
    rois = []
    if os.path.isdir(json_dir):
        for path in image_paths:
            stem = os.path.splitext(os.path.basename(path))[0]
            roi = get_cornea_circle_from_json(os.path.join(json_dir, stem + ".json"))
            if roi is not None:
                rois.append(roi)
    if not rois:
        return w / 2.0, h / 2.0, min(w, h) / 3.0
    arr = np.asarray(rois, dtype=np.float32)
    return safe_circle(float(np.median(arr[:, 0])), float(np.median(arr[:, 1])), float(np.percentile(arr[:, 2], 75)), w, h)


def load_gray_tensor(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return torch.from_numpy(img).float().unsqueeze(0) / 255.0


def crop_resize_image(path, roi, img_size, is_mask=False, roi_margin=5):
    flag = cv2.IMREAD_GRAYSCALE
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(path)
    cx, cy, r = roi
    h, w = img.shape[:2]
    x1, y1, x2, y2 = roi_crop_box(roi, w, h, roi_margin)
    crop = img[y1:y2, x1:x2]
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (img_size, img_size), interpolation=interpolation)
    return torch.from_numpy(resized).float().unsqueeze(0) / 255.0


def crop_resize_tensor(tensor, roi, img_size, is_mask=False, roi_margin=5):
    cx, cy, r = roi
    _, h, w = tensor.shape
    x1, y1, x2, y2 = roi_crop_box(roi, w, h, roi_margin)
    crop = tensor[:, y1:y2, x1:x2]
    mode = "nearest" if is_mask else "bilinear"
    kwargs = {} if is_mask else {"align_corners": False}
    return F.interpolate(crop.unsqueeze(0), size=(img_size, img_size), mode=mode, **kwargs)[0]


def roi_crop_box(roi, width, height, roi_margin=5):
    cx, cy, r = roi
    return safe_rect(cx - r - roi_margin, cy - r - roi_margin, cx + r + roi_margin, cy + r + roi_margin, width, height)


def roi_from_payload(payload):
    for shape in payload.get("shapes", []):
        if shape.get("label") != "cornea_roi":
            continue
        points = shape.get("points", [])
        if len(points) >= 2:
            cx, cy = points[0]
            px, py = points[1]
            return float(cx), float(cy), float(math.hypot(px - cx, py - cy))
    return None


def labelme_broken_mask(payload, crop_box, img_size):
    x1, y1, x2, y2 = crop_box
    scale_x = float(img_size) / max(float(x2 - x1), 1.0)
    scale_y = float(img_size) / max(float(y2 - y1), 1.0)
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    for shape in payload.get("shapes", []):
        if shape.get("label") != "broken":
            continue
        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if len(points) < 3:
            continue
        points[:, 0] = (points[:, 0] - float(x1)) * scale_x
        points[:, 1] = (points[:, 1] - float(y1)) * scale_y
        points = np.round(points).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, img_size - 1)
        points[:, 1] = np.clip(points[:, 1], 0, img_size - 1)
        cv2.fillPoly(mask, [points], 1)
    return mask


def load_patient_samples(patient_dir, window_size, img_size, target_idx):
    image_dir = os.path.join(patient_dir, "images")
    mask_dir = os.path.join(patient_dir, "mask")
    json_dir = os.path.join(patient_dir, "json")
    image_paths = list_image_paths(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {image_dir}")

    half = window_size // 2
    samples = []
    for i, image_path in enumerate(image_paths):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(json_dir, stem + ".json")
        payload = load_labelme_json(json_path) if os.path.exists(json_path) else {}
        h, w = image_hw(image_path)
        roi = roi_from_payload(payload)
        if roi is None:
            roi = get_cornea_circle_from_json(json_path) if os.path.exists(json_path) else None
        if roi is None:
            roi = estimate_patient_roi(image_paths, json_dir)
        roi = safe_circle(*roi, w, h)
        crop_box = roi_crop_box(roi, w, h)

        seq = []
        for k in range(window_size):
            j = max(0, min(i - half + k, len(image_paths) - 1))
            seq.append(crop_resize_image(image_paths[j], roi, img_size, is_mask=False))
        seq_tensor = torch.stack(seq, dim=0)
        image = normalize_image(seq_tensor[max(0, min(target_idx, window_size - 1)), 0].numpy())

        mask_path = os.path.join(mask_dir, stem + ".png")
        gt = None
        if os.path.exists(mask_path):
            gt_tensor = crop_resize_image(mask_path, roi, img_size, is_mask=True)
            gt = (gt_tensor[0].numpy() > 0.5).astype(np.uint8)
        elif payload.get("shapes"):
            gt = labelme_broken_mask(payload, crop_box, img_size)

        samples.append({
            "name": os.path.basename(image_path),
            "seq": seq_tensor,
            "image": image,
            "gt": gt,
        })
    return samples


def model_type_for(cfg, model_name):
    return cfg.get("models", {}).get(model_name, {}).get("type", "temporal")


def select_model_input(image_seq, model_type, target_idx):
    if model_type == "static":
        return image_seq[:, target_idx]
    if model_type == "temporal":
        return image_seq
    raise ValueError(f"Unsupported model type for inference: {model_type}")


def load_model_for_inference(model_name, cfg, checkpoint_dir, device, stpnet_ckpt=None):
    if model_name == "siamese_stpnet" and stpnet_ckpt:
        ckpt_path = os.path.abspath(stpnet_ckpt)
    else:
        ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"[Skip] Missing checkpoint: {ckpt_path}")
        return None
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


def run_patient_inference(args, cfg, samples, models, display_names):
    device = resolve_device(args.device)
    checkpoint_dir = os.path.abspath(args.checkpoint_dir or os.path.join(args.comparison_dir, "checkpoints"))
    target_idx = int(cfg["data"].get("target_idx", int(cfg["data"]["window_size"]) // 2))
    threshold = float(args.threshold)
    if threshold > 1.0:
        threshold /= 255.0
    use_amp = bool(cfg.get("training", {}).get("amp", True)) and device.type == "cuda"

    predictions = OrderedDict()
    available = []
    batch_size = max(1, int(args.batch_size))
    seqs = torch.stack([sample["seq"] for sample in samples], dim=0)

    if args.save_patient_predictions:
        pred_root = os.path.join(args.comparison_dir, "patient_predictions", os.path.basename(os.path.abspath(args.patient_dir)))
        os.makedirs(pred_root, exist_ok=True)
    else:
        pred_root = None

    for model_name in models:
        if model_name not in cfg.get("models", {}):
            print(f"[Skip] Model is not configured: {model_name}")
            continue
        model = load_model_for_inference(model_name, cfg, checkpoint_dir, device, stpnet_ckpt=args.stpnet_ckpt)
        if model is None:
            continue

        model_preds = []
        model_type = model_type_for(cfg, model_name)
        with torch.no_grad():
            for start in range(0, len(samples), batch_size):
                batch = seqs[start:start + batch_size].to(device)
                x = select_model_input(batch, model_type, target_idx)
                autocast_ctx = torch.amp.autocast(device_type="cuda") if use_amp else nullcontext()
                with autocast_ctx:
                    logits = get_logits(model(x))
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                for offset, item in enumerate(probs):
                    prob = item.squeeze()
                    pred = (prob > threshold).astype(np.uint8)
                    if args.debug_patient_stats:
                        sample = samples[start + offset]
                        seq_np = sample["seq"].numpy()
                        print(
                            f"[STAT] {model_name} {sample['name']} "
                            f"input=({seq_np.min():.4f},{seq_np.mean():.4f},{seq_np.max():.4f}) "
                            f"prob=({prob.min():.4f},{prob.mean():.4f},{prob.max():.4f}) "
                            f"pred_px={int(pred.sum())} gt_px={0 if sample['gt'] is None else int(sample['gt'].sum())}"
                        )
                    model_preds.append(pred)

        predictions[model_name] = model_preds
        available.append(model_name)
        print(f"[OK] Inferred {display_names.get(model_name, model_name)} on {len(samples)} frames.")

        if pred_root:
            model_dir = os.path.join(pred_root, model_name)
            os.makedirs(model_dir, exist_ok=True)
            for sample, pred in zip(samples, model_preds):
                out_name = os.path.splitext(sample["name"])[0] + ".png"
                cv2.imwrite(os.path.join(model_dir, out_name), pred.astype(np.uint8) * 255)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not available:
        raise FileNotFoundError(f"No usable checkpoints found in {checkpoint_dir}")
    return predictions, available


def build_patient_inference_figure(args):
    cfg = load_yaml_config(args.config)
    if not cfg:
        raise FileNotFoundError(f"Cannot load config: {args.config}")
    data_cfg = cfg.get("data", {})
    args.threshold = resolve_threshold(args, cfg)
    window_size = int(args.window_size or data_cfg.get("window_size", 5))
    img_size = int(args.img_size or data_cfg.get("img_size", 512))
    target_idx = int(args.target_idx if args.target_idx is not None else data_cfg.get("target_idx", window_size // 2))
    cfg.setdefault("data", {})
    cfg["data"]["window_size"] = window_size
    cfg["data"]["img_size"] = img_size
    cfg["data"]["target_idx"] = target_idx

    display_names = dict(DEFAULT_DISPLAY_NAMES)
    for name, item in cfg.get("models", {}).items():
        if args.use_config_names and isinstance(item, dict) and item.get("display_name"):
            display_names[name] = item["display_name"]

    samples = load_patient_samples(os.path.abspath(args.patient_dir), window_size, img_size, target_idx)
    indices = [idx for idx in args.indices if 0 <= idx < len(samples)]
    if not indices:
        raise ValueError("No valid patient frame indices were provided.")

    predictions, available_models = run_patient_inference(args, cfg, samples, args.models, display_names)
    has_gt = any(samples[idx]["gt"] is not None for idx in indices)
    columns = ["Input"] + (["GT"] if has_gt else []) + available_models
    n_rows, n_cols = len(indices), len(columns)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.linewidth": 0.45,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

    fig_w = max(8.0, 1.32 * n_cols)
    fig_h = max(2.0, 1.22 * n_rows + 0.55)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.patch.set_facecolor("white")

    for row, sample_idx in enumerate(indices):
        sample = samples[sample_idx]
        image = sample["image"]
        gt = sample["gt"]
        for col, column in enumerate(columns):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if column == "Input":
                ax.imshow(overlay_mask(image, None, gt if args.show_input_contour else None, alpha=args.mask_alpha))
                add_panel_label(ax, sample["name"], face="#202020")
            elif column == "GT":
                shown_gt = gt if gt is not None else np.zeros_like(image, dtype=np.uint8)
                ax.imshow(overlay_mask(image, shown_gt, None, pred_color=TRUE_SEGMENT_COLOR, alpha=0.48))
            else:
                pred = predictions[column][sample_idx]
                ax.imshow(overlay_mask(image, pred, gt, alpha=args.mask_alpha))
                if gt is not None and not args.no_dice:
                    add_panel_label(ax, f"DSC {dice_score(pred, gt):.3f}", face="#0b3c6d")

            if row == 0:
                title = column if column in {"Input", "GT"} else display_names.get(column, column)
                ax.set_title(format_column_title(title), fontsize=8.0, pad=5.5, fontweight="semibold", linespacing=0.95)

    fig.subplots_adjust(left=0.006, right=0.994, top=0.92, bottom=0.035, wspace=0.018, hspace=0.035)
    return fig


def build_patient_figure(args):
    patient_dir = os.path.abspath(args.patient_dir)
    image_dir = os.path.join(patient_dir, "images")
    json_dir = os.path.join(patient_dir, "json")
    overlay_dir = os.path.join(patient_dir, "overlays")

    image_paths = list_image_paths(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {image_dir}")

    indices = [idx for idx in args.indices if 0 <= idx < len(image_paths)]
    if not indices:
        raise ValueError("No valid patient frame indices were provided.")

    columns = ["Input", "LabelMe", "Overlay"]
    n_rows, n_cols = len(indices), len(columns)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.linewidth": 0.45,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

    fig_w = 7.5
    fig_h = max(2.2, 1.65 * n_rows + 0.55)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.patch.set_facecolor("white")

    for row, idx in enumerate(indices):
        image_path = image_paths[idx]
        frame_name = os.path.basename(image_path)
        stem = os.path.splitext(frame_name)[0]
        json_path = os.path.join(json_dir, stem + ".json")
        payload = load_labelme_json(json_path)

        img_rgb = read_rgb_image(image_path)
        base = image_to_display_rgb(img_rgb)
        labelme_img = draw_labelme_shapes(base, payload, alpha=args.mask_alpha)

        overlay_path = os.path.join(overlay_dir, frame_name)
        if os.path.exists(overlay_path):
            overlay_img = read_rgb_image(overlay_path).astype(np.float32) / 255.0
        else:
            overlay_img = labelme_img

        panels = [base, labelme_img, overlay_img]
        for col, (title, panel) in enumerate(zip(columns, panels)):
            ax = axes[row, col]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col == 0:
                add_panel_label(ax, frame_name, face="#202020")
            elif col == 1:
                add_panel_label(ax, f"broken {count_broken(payload)}", face="#8b1d1d")
            if row == 0:
                ax.set_title(title, fontsize=9, pad=5.5, fontweight="semibold")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.035, wspace=0.025, hspace=0.055)
    return fig


def load_predictions(comparison_dir, models, sample_idx, shape, threshold):
    pred_root = os.path.join(comparison_dir, "predictions")
    preds = OrderedDict()
    for model in models:
        path = os.path.join(pred_root, model, f"pred_{sample_idx:04d}.png")
        pred = read_pred_mask(path, threshold)
        preds[model] = resize_mask(pred, shape) if pred is not None else None
    return preds


def add_panel_label(ax, text, face="#111111"):
    ax.text(
        0.02,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="white",
        bbox=dict(boxstyle="round,pad=0.20", facecolor=face, edgecolor="none", alpha=0.82),
    )


def format_column_title(title, width=13):
    if len(title) <= width:
        return title
    return "\n".join(textwrap.wrap(title, width=width, break_long_words=False))


def build_figure(args):
    val_dir, window_size, img_size, target_idx, display_names = resolve_settings(args)
    dataset = SequencePMTMDataset(
        root_dir=val_dir,
        window_size=window_size,
        img_size=img_size,
        is_train=False,
    )

    available_models = []
    for model in args.models:
        pred_dir = os.path.join(args.comparison_dir, "predictions", model)
        if os.path.isdir(pred_dir):
            available_models.append(model)
        else:
            print(f"[Skip] Missing prediction directory: {pred_dir}")

    if not available_models:
        raise FileNotFoundError(f"No prediction directories found under {args.comparison_dir!r}.")

    indices = [idx for idx in args.indices if 0 <= idx < len(dataset)]
    if not indices:
        raise ValueError("No valid sample indices were provided.")

    columns = ["Input", "GT"] + available_models
    n_rows, n_cols = len(indices), len(columns)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.linewidth": 0.45,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

    fig_w = max(8.0, 1.32 * n_cols)
    fig_h = max(2.0, 1.22 * n_rows + 0.55)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.patch.set_facecolor("white")

    for row, sample_idx in enumerate(indices):
        seq_tensor, mask_tensor = dataset[sample_idx]
        image = tensor_to_image(seq_tensor, target_idx)
        gt = tensor_to_mask(mask_tensor)
        preds = load_predictions(args.comparison_dir, available_models, sample_idx, gt.shape, args.threshold)

        for col, column in enumerate(columns):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if column == "Input":
                shown = overlay_mask(
                    image,
                    None,
                    gt if args.show_input_contour else None,
                    alpha=args.mask_alpha,
                )
                ax.imshow(shown)
                add_panel_label(ax, f"#{sample_idx:02d}", face="#202020")
            elif column == "GT":
                ax.imshow(overlay_mask(image, gt, None, pred_color=TRUE_SEGMENT_COLOR, alpha=0.48))
            else:
                pred = preds[column]
                ax.imshow(overlay_mask(image, pred, gt, alpha=args.mask_alpha))
                if not args.no_dice:
                    dsc = dice_score(pred, gt)
                    label = "N/A" if math.isnan(dsc) else f"DSC {dsc:.3f}"
                    add_panel_label(ax, label, face="#0b3c6d")

            if row == 0:
                title = column if column in {"Input", "GT"} else display_names.get(column, column)
                ax.set_title(format_column_title(title), fontsize=8.0, pad=5.5, fontweight="semibold", linespacing=0.95)

    fig.subplots_adjust(left=0.006, right=0.994, top=0.92, bottom=0.035, wspace=0.018, hspace=0.035)
    return fig


def main():
    args = parse_args()
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if args.patient_dir and args.patient_mode == "inference":
        fig = build_patient_inference_figure(args)
    elif args.patient_dir:
        fig = build_patient_figure(args)
    else:
        fig = build_figure(args)
    fig.savefig(args.output, dpi=args.dpi)
    print(f"Wrote {args.output}")

    if args.pdf:
        pdf_dir = os.path.dirname(args.pdf)
        if pdf_dir:
            os.makedirs(pdf_dir, exist_ok=True)
        fig.savefig(args.pdf)
        print(f"Wrote {args.pdf}")

    plt.close(fig)


if __name__ == "__main__":
    main()

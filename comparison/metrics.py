import math
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
except Exception:  # pragma: no cover - fallback only used when scipy is absent.
    binary_erosion = None
    distance_transform_edt = None

from comparison.losses import extract_main_logits


class SegmentationMetricAccumulator:
    def __init__(self, threshold=0.5, eps=1e-7):
        self.threshold = threshold
        self.eps = eps
        self.tp = 0.0
        self.fp = 0.0
        self.tn = 0.0
        self.fn = 0.0
        self.hd95_values = []

    def update(self, logits, targets):
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        targets = (targets > 0.5).float()

        self.tp += float((preds * targets).sum().item())
        self.fp += float((preds * (1.0 - targets)).sum().item())
        self.fn += float(((1.0 - preds) * targets).sum().item())
        self.tn += float(((1.0 - preds) * (1.0 - targets)).sum().item())

        pred_np = preds.detach().cpu().numpy().astype(bool)
        target_np = targets.detach().cpu().numpy().astype(bool)
        for p, t in zip(pred_np, target_np):
            self.hd95_values.append(hd95(p.squeeze(), t.squeeze()))

    def compute(self):
        tp, fp, tn, fn, eps = self.tp, self.fp, self.tn, self.fn, self.eps
        dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        specificity = (tn + eps) / (tn + fp + eps)
        hd95_value = float(np.mean(self.hd95_values)) if self.hd95_values else 0.0
        return {
            "Dice": dice,
            "Precision": precision,
            "Recall": recall,
            "Specificity": specificity,
            "HD95": hd95_value,
        }


def hd95(pred, target):
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    if not pred.any() and not target.any():
        return 0.0
    if pred.any() != target.any():
        # One empty mask has no surface. Use the image diagonal as a bounded
        # large penalty so metric export never crashes.
        return float(math.hypot(pred.shape[-2], pred.shape[-1]))
    if binary_erosion is None or distance_transform_edt is None:
        return float("nan")

    pred_surface = pred ^ binary_erosion(pred)
    target_surface = target ^ binary_erosion(target)
    if not pred_surface.any() or not target_surface.any():
        return float(math.hypot(pred.shape[-2], pred.shape[-1]))

    dt_pred = distance_transform_edt(~pred_surface)
    dt_target = distance_transform_edt(~target_surface)
    distances = np.concatenate([dt_target[pred_surface], dt_pred[target_surface]])
    return float(np.percentile(distances, 95)) if distances.size else 0.0


def count_parameters_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def estimate_flops_g(model, input_tensor, device):
    was_training = model.training
    model.eval()
    hooks = []
    flops = {"value": 0.0}

    def conv_hook(module, inputs, output):
        x = inputs[0]
        if not torch.is_tensor(output):
            return
        out = output
        out_h, out_w = out.shape[-2:]
        batch = out.shape[0]
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels / module.groups)
        flops["value"] += batch * out_h * out_w * module.out_channels * kernel_ops

    def linear_hook(module, inputs, output):
        batch_ops = int(np.prod(output.shape[:-1])) if output.dim() > 1 else 1
        flops["value"] += batch_ops * module.in_features * module.out_features

    def bn_hook(module, inputs, output):
        flops["value"] += output.numel()

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.GroupNorm, torch.nn.LayerNorm)):
            hooks.append(module.register_forward_hook(bn_hook))

    try:
        with torch.no_grad():
            model(input_tensor.to(device))
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    return flops["value"] / 1e9


def measure_fps(model, input_tensor, device, warmup=5, iterations=20):
    was_training = model.training
    model.eval()
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    model.train(was_training)
    return float(iterations * input_tensor.shape[0] / max(elapsed, 1e-9))


def get_logits(outputs):
    return extract_main_logits(outputs)


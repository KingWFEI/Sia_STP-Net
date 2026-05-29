import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_main_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["seg"]
    return outputs


class DiceProbLoss(nn.Module):
    def forward(self, probs, targets, eps=1e-7):
        probs = probs.float()
        targets = targets.float()
        dims = tuple(range(1, probs.dim()))
        inter = (probs * targets).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets.sum(dim=dims)
        return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


class UnifiedSegLoss(nn.Module):
    """BCEWithLogits + Dice loss adapter for tensor and dict model outputs."""

    def __init__(
        self,
        pos_weight=2.0,
        lambda_bce=1.0,
        lambda_dice=1.0,
        aux_weights=None,
    ):
        super().__init__()
        pos_weight = torch.tensor([float(pos_weight)]) if not torch.is_tensor(pos_weight) else pos_weight.float()
        self.register_buffer("pos_weight", pos_weight)
        self.lambda_bce = float(lambda_bce)
        self.lambda_dice = float(lambda_dice)
        self.aux_weights = list(aux_weights or [0.4, 0.3, 0.2])
        self.dice = DiceProbLoss()

    def _single_loss(self, logits, targets):
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=self.pos_weight)
        dice = self.dice(torch.sigmoid(logits), targets)
        return self.lambda_bce * bce + self.lambda_dice * dice

    def forward(self, outputs, targets):
        if not isinstance(outputs, dict):
            return self._single_loss(outputs, targets)

        loss = self._single_loss(outputs["seg"], targets)
        for i, aux_logits in enumerate(outputs.get("aux", [])):
            weight = self.aux_weights[i] if i < len(self.aux_weights) else self.aux_weights[-1]
            loss = loss + float(weight) * self._single_loss(aux_logits, targets)
        return loss


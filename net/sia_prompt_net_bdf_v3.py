import torch
import torch.nn as nn

from net.sia_prompt_net_bdf import STNFeatureAligner
from net.sia_prompt_net_bdf import SiameseSTPromptNet as BaseSiameseSTPromptNet


class BiGatedDifferenceFusionV3(nn.Module):
    """BDF-v3: add target_project before difference evidence construction."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.aligner = STNFeatureAligner(channels)
        self.target_project = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=channels),
            nn.ReLU(inplace=True),
        )

        in_ch = channels * 4 + 1
        self.phi_evidence = nn.Sequential(
            nn.Conv2d(in_ch, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=channels),
            nn.ReLU(inplace=True),
        )
        self.psi_reliability = nn.Conv2d(channels, 1, kernel_size=1)
        self.alpha_fwd = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.alpha_bwd = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.fusion_project = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=channels),
            nn.ReLU(inplace=True),
        )

    def _relative_position_map(self, rel_idx, batch_size, height, width, device, dtype, norm_factor):
        rel_value = rel_idx / float(max(norm_factor, 1))
        return torch.full((batch_size, 1, height, width), rel_value, device=device, dtype=dtype)

    def forward(self, feature_sequence):
        b, _, h, w = feature_sequence[0].shape
        t_total = len(feature_sequence)
        target_idx = t_total // 2
        feat_target_raw = feature_sequence[target_idx]
        feat_target = self.target_project(feat_target_raw)
        norm_factor = max(target_idx, t_total - 1 - target_idx, 1)

        aligned_seq = []
        for t in range(t_total):
            if t == target_idx:
                aligned_seq.append(feat_target)
            else:
                aligned_seq.append(self.aligner(feature_sequence[t], feat_target))

        e_hat_seq = []
        gate_maps = []
        for t in range(t_total):
            if t == target_idx:
                e_hat_seq.append(feat_target)
                gate_maps.append(torch.ones((b, 1, h, w), device=feat_target.device, dtype=feat_target.dtype))
                continue

            feat_t = aligned_seq[t]
            diff = feat_t - feat_target
            abs_diff = diff.abs()
            pe_map = self._relative_position_map(
                t - target_idx, b, h, w, feat_target.device, feat_target.dtype, norm_factor
            )
            descriptor = torch.cat([feat_t, feat_target, diff, abs_diff, pe_map], dim=1)
            e_t = self.phi_evidence(descriptor)
            r_t = torch.sigmoid(self.psi_reliability(e_t))
            e_hat_seq.append(r_t * e_t)
            gate_maps.append(r_t)

        s_fwd = torch.zeros_like(feat_target)
        s_bwd = torch.zeros_like(feat_target)
        s_fwd_list = [None] * t_total
        s_bwd_list = [None] * t_total

        for t in range(t_total):
            e_hat_t = e_hat_seq[t]
            alpha = torch.sigmoid(self.alpha_fwd(torch.cat([e_hat_t, s_fwd], dim=1)))
            s_fwd = alpha * s_fwd + (1.0 - alpha) * e_hat_t
            s_fwd_list[t] = s_fwd

        for t in range(t_total - 1, -1, -1):
            e_hat_t = e_hat_seq[t]
            alpha = torch.sigmoid(self.alpha_bwd(torch.cat([e_hat_t, s_bwd], dim=1)))
            s_bwd = alpha * s_bwd + (1.0 - alpha) * e_hat_t
            s_bwd_list[t] = s_bwd

        merged_state = self.fusion_project(torch.cat([s_fwd_list[target_idx], s_bwd_list[target_idx]], dim=1))
        return {
            "prompt": merged_state + feat_target,
            "gate_maps": torch.stack(gate_maps, dim=1),
        }


class SiameseSTPromptNetBDFV3(BaseSiameseSTPromptNet):
    def __init__(self, num_classes, input_channels=1, deep_supervision=True):
        super().__init__(num_classes=num_classes, input_channels=input_channels, deep_supervision=deep_supervision)
        self.st_fusion_bottleneck = BiGatedDifferenceFusionV3(512)


def build_model(window_size=5, image_size=(512, 512), num_classes=1, input_channels=1):
    print(f"[Model] Building Siamese ST-Prompt-Net BDF-v3 | Window: {window_size}, Input: {image_size}")
    return SiameseSTPromptNetBDFV3(
        num_classes=num_classes,
        input_channels=input_channels,
        deep_supervision=True,
    )


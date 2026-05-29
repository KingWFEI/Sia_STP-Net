from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
#  1. Basic Blocks
# =========================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(mid_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1, x2, x3, x4 = self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)
        x5 = F.interpolate(self.pool(x), size=x.shape[2:], mode="bilinear", align_corners=True)
        return self.project(torch.cat([x1, x2, x3, x4, x5], dim=1))


class SiameseEncoderWithASPP(nn.Module):
    def __init__(self, input_channels, nb_filter):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv0_0 = ConvBlock(input_channels, nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4])
        self.aspp = ASPP(nb_filter[4], nb_filter[4])

    def forward(self, x):
        x0 = self.conv0_0(x)
        x1 = self.conv1_0(self.pool(x0))
        x2 = self.conv2_0(self.pool(x1))
        x3 = self.conv3_0(self.pool(x2))
        x4 = self.aspp(self.conv4_0(self.pool(x3)))
        return [x0, x1, x2, x3, x4]


# =========================================================================
#  2. Temporal / Alignment Blocks
# =========================================================================
class STNFeatureAligner(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(channels * 2, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 16, 128),
            nn.ReLU(True),
            nn.Linear(128, 6),
        )
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, feat_moving, feat_fixed):
        theta = self.localization(torch.cat([feat_moving, feat_fixed], dim=1)).view(-1, 2, 3)
        grid = F.affine_grid(theta, feat_moving.size(), align_corners=True)
        return F.grid_sample(feat_moving, grid, align_corners=True)


class TemporalMaskHead(nn.Module):
    def __init__(self, in_ch, out_ch=1):
        super().__init__()
        mid_ch = max(in_ch // 4, 32)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        cc_i, cc_f, cc_o, cc_g = torch.split(self.conv(combined), self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        h, w = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, h, w, device=device),
            torch.zeros(batch_size, self.hidden_dim, h, w, device=device),
        )


# =========================================================================
#  3. Deep Fusion Modules for Ablation
# =========================================================================
class IdentityFusion(nn.Module):
    """Ablation: no deep temporal fusion, use target x4 directly."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, feature_sequence):
        target_idx = len(feature_sequence) // 2
        feat_target = feature_sequence[target_idx]
        B, C, H, W = feat_target.shape
        T = len(feature_sequence)
        aligned_seq = torch.stack(feature_sequence, dim=1)
        gate_maps = torch.ones(B, T, 1, H, W, device=feat_target.device, dtype=feat_target.dtype)
        return {
            "prompt": feat_target,
            "gate_maps": gate_maps,
            "aligned_seq": aligned_seq,
        }


class STNMeanFusion(nn.Module):
    """
    Ablation for explicit alignment only.
    Note: your original file does not contain a standalone "STN-only" fusion head,
    so this lightweight aligned-mean aggregator is added only for ablation isolation.
    """
    def __init__(self, channels):
        super().__init__()
        self.aligner = STNFeatureAligner(channels)

    def forward(self, feature_sequence):
        target_idx = len(feature_sequence) // 2
        feat_target = feature_sequence[target_idx]
        aligned_seq = []
        for t, feat_t in enumerate(feature_sequence):
            if t == target_idx:
                aligned_seq.append(feat_target)
            else:
                aligned_seq.append(self.aligner(feat_t, feat_target))

        aligned_stack = torch.stack(aligned_seq, dim=1)
        mean_aligned = torch.mean(aligned_stack, dim=1)
        prompt = 0.5 * (feat_target + mean_aligned)
        B, _, H, W = feat_target.shape
        T = len(feature_sequence)
        gate_maps = torch.ones(B, T, 1, H, W, device=feat_target.device, dtype=feat_target.dtype)
        return {
            "prompt": prompt,
            "gate_maps": gate_maps,
            "aligned_seq": aligned_stack,
        }


class BiGatedDifferenceFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.aligner = STNFeatureAligner(channels)

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
        B, C, H, W = feature_sequence[0].shape
        T = len(feature_sequence)
        target_idx = T // 2
        feat_target = feature_sequence[target_idx]
        norm_factor = max(target_idx, T - 1 - target_idx, 1)

        aligned_seq = []
        for t in range(T):
            if t == target_idx:
                aligned_seq.append(feat_target)
            else:
                aligned_seq.append(self.aligner(feature_sequence[t], feat_target))

        e_hat_seq = []
        gate_maps_rt = []
        for t in range(T):
            if t == target_idx:
                e_hat_seq.append(feat_target)
                gate_maps_rt.append(torch.ones((B, 1, H, W), device=feat_target.device, dtype=feat_target.dtype))
                continue

            feat_t = aligned_seq[t]
            diff = feat_t - feat_target
            abs_diff = diff.abs()
            pe_map = self._relative_position_map(
                t - target_idx, B, H, W, feat_target.device, feat_target.dtype, norm_factor
            )
            descriptor = torch.cat([feat_t, feat_target, diff, abs_diff, pe_map], dim=1)
            e_t = self.phi_evidence(descriptor)
            r_t = torch.sigmoid(self.psi_reliability(e_t))
            e_hat_seq.append(r_t * e_t)
            gate_maps_rt.append(r_t)

        s_fwd = torch.zeros_like(feat_target)
        s_bwd = torch.zeros_like(feat_target)
        s_fwd_list = [None] * T
        s_bwd_list = [None] * T

        for t in range(T):
            e_hat_t = e_hat_seq[t]
            alpha_fwd_t = torch.sigmoid(self.alpha_fwd(torch.cat([e_hat_t, s_fwd], dim=1)))
            s_fwd = alpha_fwd_t * s_fwd + (1.0 - alpha_fwd_t) * e_hat_t
            s_fwd_list[t] = s_fwd

        for t in range(T - 1, -1, -1):
            e_hat_t = e_hat_seq[t]
            alpha_bwd_t = torch.sigmoid(self.alpha_bwd(torch.cat([e_hat_t, s_bwd], dim=1)))
            s_bwd = alpha_bwd_t * s_bwd + (1.0 - alpha_bwd_t) * e_hat_t
            s_bwd_list[t] = s_bwd

        fwd_target_state = s_fwd_list[target_idx]
        bwd_target_state = s_bwd_list[target_idx]
        merged_state = self.fusion_project(torch.cat([fwd_target_state, bwd_target_state], dim=1))
        prompt = merged_state + feat_target

        return {
            "prompt": prompt,
            "gate_maps": torch.stack(gate_maps_rt, dim=1),
            "aligned_seq": torch.stack(aligned_seq, dim=1),
        }


class BiConvLSTMFusion(nn.Module):
    """
    Bi-ConvLSTM ablation version built on top of your current file.
    If you have a historical exact Bi-ConvLSTM implementation, prefer that for final comparison.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.aligner = STNFeatureAligner(channels)
        self.fwd_cell = ConvLSTMCell(channels, channels, kernel_size=3, bias=True)
        self.bwd_cell = ConvLSTMCell(channels, channels, kernel_size=3, bias=True)
        self.fusion_project = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature_sequence):
        B, C, H, W = feature_sequence[0].shape
        T = len(feature_sequence)
        target_idx = T // 2
        feat_target = feature_sequence[target_idx]

        aligned_seq = []
        for t in range(T):
            if t == target_idx:
                aligned_seq.append(feat_target)
            else:
                aligned_seq.append(self.aligner(feature_sequence[t], feat_target))

        h_fwd, c_fwd = self.fwd_cell.init_hidden(B, (H, W), feat_target.device)
        h_bwd, c_bwd = self.bwd_cell.init_hidden(B, (H, W), feat_target.device)
        fwd_states = [None] * T
        bwd_states = [None] * T

        for t in range(T):
            h_fwd, c_fwd = self.fwd_cell(aligned_seq[t], (h_fwd, c_fwd))
            fwd_states[t] = h_fwd

        for t in range(T - 1, -1, -1):
            h_bwd, c_bwd = self.bwd_cell(aligned_seq[t], (h_bwd, c_bwd))
            bwd_states[t] = h_bwd

        prompt = self.fusion_project(torch.cat([fwd_states[target_idx], bwd_states[target_idx]], dim=1)) + feat_target
        gate_maps = torch.ones(B, T, 1, H, W, device=feat_target.device, dtype=feat_target.dtype)
        return {
            "prompt": prompt,
            "gate_maps": gate_maps,
            "aligned_seq": torch.stack(aligned_seq, dim=1),
        }


# =========================================================================
#  4. Decoder Blocks
# =========================================================================
class TemporalPromptAttention(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, prompt, skip_feat):
        g1 = self.W_g(prompt)
        x1 = self.W_x(skip_feat)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)
        psi = self.relu(g1 + x1)
        attention_mask = self.psi(psi)
        return skip_feat * attention_mask


class PromptDecoderBlock(nn.Module):
    def __init__(self, prompt_ch, skip_ch, out_ch):
        super().__init__()
        self.attention = TemporalPromptAttention(F_g=prompt_ch, F_l=skip_ch, F_int=max(skip_ch // 2, 16))
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(prompt_ch + skip_ch, out_ch)

    def forward(self, prompt, skip):
        activated_skip = self.attention(prompt, skip)
        prompt_up = self.up(prompt)
        return self.conv(torch.cat([prompt_up, activated_skip], dim=1))


class VanillaDecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x_up = self.up(x)
        return self.conv(torch.cat([x_up, skip], dim=1))


# =========================================================================
#  5. Ablation Config + Main Model
# =========================================================================
@dataclass
class AblationConfig:
    use_shallow_temporal: bool = True
    deep_fusion: str = "bdf"  # one of: none, stn_mean, bdf, biconvlstm
    use_prior_decoder: bool = True
    use_temporal_mask_head: bool = False


class SiameseSTPromptNetAblation(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, deep_supervision=True, config: Optional[AblationConfig] = None):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.config = config or AblationConfig()
        nb_filter = [32, 64, 128, 256, 512]

        self.encoder = SiameseEncoderWithASPP(input_channels, nb_filter)

        self.temporal_extractor_0 = nn.Sequential(
            nn.Conv3d(nb_filter[0], nb_filter[0], kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=nb_filter[0], bias=False),
            nn.BatchNorm3d(nb_filter[0]),
            nn.ReLU(inplace=True),
        )
        self.temporal_extractor_1 = nn.Sequential(
            nn.Conv3d(nb_filter[1], nb_filter[1], kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=nb_filter[1], bias=False),
            nn.BatchNorm3d(nb_filter[1]),
            nn.ReLU(inplace=True),
        )

        if self.config.deep_fusion == "none":
            self.deep_fusion = IdentityFusion(nb_filter[4])
        elif self.config.deep_fusion == "stn_mean":
            self.deep_fusion = STNMeanFusion(nb_filter[4])
        elif self.config.deep_fusion == "bdf":
            self.deep_fusion = BiGatedDifferenceFusion(nb_filter[4])
        elif self.config.deep_fusion == "biconvlstm":
            self.deep_fusion = BiConvLSTMFusion(nb_filter[4])
        else:
            raise ValueError(f"Unsupported deep_fusion: {self.config.deep_fusion}")

        if self.config.use_temporal_mask_head:
            self.temporal_mask_head = TemporalMaskHead(in_ch=nb_filter[4], out_ch=num_classes)
        else:
            self.temporal_mask_head = None

        if self.config.use_prior_decoder:
            self.dec3 = PromptDecoderBlock(prompt_ch=nb_filter[4], skip_ch=nb_filter[3], out_ch=nb_filter[3])
            self.dec2 = PromptDecoderBlock(prompt_ch=nb_filter[3], skip_ch=nb_filter[2], out_ch=nb_filter[2])
            self.dec1 = PromptDecoderBlock(prompt_ch=nb_filter[2], skip_ch=nb_filter[1], out_ch=nb_filter[1])
            self.dec0 = PromptDecoderBlock(prompt_ch=nb_filter[1], skip_ch=nb_filter[0], out_ch=nb_filter[0])
        else:
            self.dec3 = VanillaDecoderBlock(in_ch=nb_filter[4], skip_ch=nb_filter[3], out_ch=nb_filter[3])
            self.dec2 = VanillaDecoderBlock(in_ch=nb_filter[3], skip_ch=nb_filter[2], out_ch=nb_filter[2])
            self.dec1 = VanillaDecoderBlock(in_ch=nb_filter[2], skip_ch=nb_filter[1], out_ch=nb_filter[1])
            self.dec0 = VanillaDecoderBlock(in_ch=nb_filter[1], skip_ch=nb_filter[0], out_ch=nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, 1)
        self.final1 = nn.Conv2d(nb_filter[1], num_classes, 1)
        self.final2 = nn.Conv2d(nb_filter[2], num_classes, 1)
        self.final3 = nn.Conv2d(nb_filter[3], num_classes, 1)

    def _extract_target_frame(self, feat_flat, B, T, target_idx):
        _, C, H, W = feat_flat.shape
        return feat_flat.view(B, T, C, H, W)[:, target_idx, :, :, :]

    def _extract_sequence(self, feat_flat, B, T):
        _, C, H, W = feat_flat.shape
        feat_seq_tensor = feat_flat.view(B, T, C, H, W)
        return [feat_seq_tensor[:, t, :, :, :] for t in range(T)]

    def _apply_temporal_filter(self, feat_flat, B, T, target_idx, filter_module):
        _, C, H, W = feat_flat.shape
        feat_seq = feat_flat.view(B, T, C, H, W)
        feat_seq_3d = feat_seq.permute(0, 2, 1, 3, 4)
        filtered_seq_3d = filter_module(feat_seq_3d)
        return filtered_seq_3d[:, :, target_idx, :, :]

    def forward(self, x_sequence):
        B, T, C, H, W = x_sequence.shape
        target_idx = T // 2
        x_reshaped = x_sequence.view(B * T, C, H, W)

        x0_flat, x1_flat, x2_flat, x3_flat, x4_flat = self.encoder(x_reshaped)

        if self.config.use_shallow_temporal:
            x0_tgt = self._apply_temporal_filter(x0_flat, B, T, target_idx, self.temporal_extractor_0)
            x1_tgt = self._apply_temporal_filter(x1_flat, B, T, target_idx, self.temporal_extractor_1)
        else:
            x0_tgt = self._extract_target_frame(x0_flat, B, T, target_idx)
            x1_tgt = self._extract_target_frame(x1_flat, B, T, target_idx)

        x2_tgt = self._extract_target_frame(x2_flat, B, T, target_idx)
        x3_tgt = self._extract_target_frame(x3_flat, B, T, target_idx)

        x4_seq = self._extract_sequence(x4_flat, B, T)
        bottleneck_out = self.deep_fusion(x4_seq)
        bottleneck = bottleneck_out["prompt"]

        temporal_seq_logits = None
        if self.training and self.temporal_mask_head is not None:
            aligned_seq = bottleneck_out["aligned_seq"]
            B1, T1, C1, Hb, Wb = aligned_seq.shape
            aligned_seq_flat = aligned_seq.view(B1 * T1, C1, Hb, Wb)
            temporal_seq_logits = self.temporal_mask_head(aligned_seq_flat).view(B1, T1, -1, Hb, Wb)

        if self.config.use_prior_decoder:
            d3 = self.dec3(prompt=bottleneck, skip=x3_tgt)
            d2 = self.dec2(prompt=d3, skip=x2_tgt)
            d1 = self.dec1(prompt=d2, skip=x1_tgt)
            d0 = self.dec0(prompt=d1, skip=x0_tgt)
        else:
            d3 = self.dec3(x=bottleneck, skip=x3_tgt)
            d2 = self.dec2(x=d3, skip=x2_tgt)
            d1 = self.dec1(x=d2, skip=x1_tgt)
            d0 = self.dec0(x=d1, skip=x0_tgt)

        output = self.final(d0)

        if self.training:
            out_dict: Dict[str, torch.Tensor] = {
                "seg": output,
                "gate_maps": bottleneck_out["gate_maps"],
            }
            if temporal_seq_logits is not None:
                out_dict["temporal_seq_logits"] = temporal_seq_logits
            if self.deep_supervision:
                target_size = output.shape[2:]
                aux1 = F.interpolate(self.final1(d1), size=target_size, mode="bilinear", align_corners=True)
                aux2 = F.interpolate(self.final2(d2), size=target_size, mode="bilinear", align_corners=True)
                aux3 = F.interpolate(self.final3(d3), size=target_size, mode="bilinear", align_corners=True)
                out_dict["aux"] = [aux1, aux2, aux3]
            return out_dict

        return output


# =========================================================================
#  6. Build Interface for Ablation
# =========================================================================
ABLATION_PRESETS = {
    "baseline": AblationConfig(
        use_shallow_temporal=False,
        deep_fusion="none",
        use_prior_decoder=False,
        use_temporal_mask_head=False,
    ),
    "shallow_temporal": AblationConfig(
        use_shallow_temporal=True,
        deep_fusion="none",
        use_prior_decoder=False,
        use_temporal_mask_head=False,
    ),
    "stn_mean": AblationConfig(
        use_shallow_temporal=True,
        deep_fusion="stn_mean",
        use_prior_decoder=False,
        use_temporal_mask_head=False,
    ),
    "bdf_no_prior": AblationConfig(
        use_shallow_temporal=True,
        deep_fusion="bdf",
        use_prior_decoder=False,
        use_temporal_mask_head=False,
    ),
    "full": AblationConfig(
        use_shallow_temporal=True,
        deep_fusion="bdf",
        use_prior_decoder=True,
        use_temporal_mask_head=False,
    ),
    "biconvlstm": AblationConfig(
        use_shallow_temporal=True,
        deep_fusion="biconvlstm",
        use_prior_decoder=True,
        use_temporal_mask_head=False,
    ),
}


def build_model(
    window_size=5,
    image_size=(256, 256),
    num_classes=1,
    input_channels=1,
    ablation_name="full",
    deep_supervision=True,
):
    if ablation_name not in ABLATION_PRESETS:
        raise ValueError(f"Unsupported ablation_name: {ablation_name}. Available: {list(ABLATION_PRESETS.keys())}")

    cfg = ABLATION_PRESETS[ablation_name]
    print(
        f"[Model] Build Siamese STP-Net Ablation | preset={ablation_name} | "
        f"window={window_size} | image_size={image_size}"
    )
    print(f"[Ablation Config] {cfg}")

    model = SiameseSTPromptNetAblation(
        num_classes=num_classes,
        input_channels=input_channels,
        deep_supervision=deep_supervision,
        config=cfg,
    )
    return model


if __name__ == "__main__":
    x = torch.randn(2, 5, 1, 512, 512)
    for name in ABLATION_PRESETS.keys():
        model = build_model(window_size=5, image_size=(512, 512), num_classes=1, input_channels=1, ablation_name=name)
        model.train()
        y = model(x)
        print(name, y["seg"].shape)

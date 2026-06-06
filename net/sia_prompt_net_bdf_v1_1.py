import torch
import torch.nn as nn
import torch.nn.functional as F

from net.sia_prompt_net_bdf import ASPP
from net.sia_prompt_net_bdf import BiGatedDifferenceFusion
from net.sia_prompt_net_bdf import ConvBlock
from net.sia_prompt_net_bdf import PromptDecoderBlock
from net.sia_prompt_net_bdf import SiameseEncoderWithASPP
from net.sia_prompt_net_bdf import SiameseSTPromptNet as BaseSiameseSTPromptNet
from net.sia_prompt_net_bdf import STNFeatureAligner
from net.sia_prompt_net_bdf import TemporalPromptAttention


def _make_gn(num_channels, max_groups=16):
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(num_groups=groups, num_channels=num_channels)


class ShallowReliabilityGate(nn.Module):
    def __init__(self, prompt_ch, skip_ch, hidden_ch):
        super().__init__()
        self.prompt_ch = prompt_ch
        self.skip_ch = skip_ch
        self.hidden_ch = hidden_ch
        self.proj = nn.Sequential(
            nn.Conv2d(prompt_ch, hidden_ch, kernel_size=1, bias=False),
            _make_gn(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, 1, kernel_size=1, bias=True),
        )
        nn.init.constant_(self.proj[-1].bias, -2.0)

    def forward(self, prompt, skip_tgt):
        gate_logits = self.proj(prompt)
        gate_logits = F.interpolate(
            gate_logits,
            size=skip_tgt.shape[2:],
            mode="bilinear",
            align_corners=True,
        )
        return torch.sigmoid(gate_logits)


class SiameseSTPromptNet(BaseSiameseSTPromptNet):
    def __init__(self, num_classes, input_channels=1, deep_supervision=True):
        super().__init__(
            num_classes=num_classes,
            input_channels=input_channels,
            deep_supervision=deep_supervision,
        )
        nb_filter = [32, 64, 128, 256, 512]
        self.skip_gate_1 = ShallowReliabilityGate(
            prompt_ch=nb_filter[4],
            skip_ch=nb_filter[1],
            hidden_ch=nb_filter[1],
        )
        self.skip_gate_0 = ShallowReliabilityGate(
            prompt_ch=nb_filter[4],
            skip_ch=nb_filter[0],
            hidden_ch=nb_filter[0],
        )

    def forward(self, x_sequence):
        B, T, C, H, W = x_sequence.shape
        target_idx = T // 2
        x_reshaped = x_sequence.view(B * T, C, H, W)

        enc_feats_flat = self.encoder(x_reshaped)
        x0_flat, x1_flat, x2_flat, x3_flat, x4_flat = enc_feats_flat

        x0_enriched_raw = self._apply_temporal_filter(
            x0_flat, B, T, target_idx, self.temporal_extractor_0
        )
        x1_enriched_raw = self._apply_temporal_filter(
            x1_flat, B, T, target_idx, self.temporal_extractor_1
        )

        x0_tgt = self._extract_target_frame(x0_flat, B, T, target_idx)
        x1_tgt = self._extract_target_frame(x1_flat, B, T, target_idx)
        x2_tgt = self._extract_target_frame(x2_flat, B, T, target_idx)
        x3_tgt = self._extract_target_frame(x3_flat, B, T, target_idx)

        x4_seq = self._extract_sequence(x4_flat, B, T)
        bottleneck_out = self.st_fusion_bottleneck(x4_seq)

        temporal_prompt_4 = bottleneck_out["prompt"]
        gate_maps = bottleneck_out["gate_maps"]

        gate1_skip = self.skip_gate_1(temporal_prompt_4, x1_tgt)
        gate0_skip = self.skip_gate_0(temporal_prompt_4, x0_tgt)
        x1_enriched = x1_tgt + gate1_skip * (x1_enriched_raw - x1_tgt)
        x0_enriched = x0_tgt + gate0_skip * (x0_enriched_raw - x0_tgt)

        d3 = self.dec3(prompt=temporal_prompt_4, skip=x3_tgt)
        d2 = self.dec2(prompt=d3, skip=x2_tgt)
        d1 = self.dec1(prompt=d2, skip=x1_enriched)
        d0 = self.dec0(prompt=d1, skip=x0_enriched)

        output = self.final(d0)

        if self.training or self.return_diagnostics:
            out_dict = {
                "seg": output,
                "gate_maps": gate_maps,
                "prompt": temporal_prompt_4,
                "target_feat": bottleneck_out.get("target_feat"),
                "merged_state": bottleneck_out.get("merged_state"),
                "aligned_seq": bottleneck_out.get("aligned_seq"),
                "gate0_skip": gate0_skip,
                "gate1_skip": gate1_skip,
                "x0_enriched_raw": x0_enriched_raw,
                "x1_enriched_raw": x1_enriched_raw,
                "x0_tgt": x0_tgt,
                "x1_tgt": x1_tgt,
            }

            if self.training and self.deep_supervision:
                target_size = output.shape[2:]
                aux1 = F.interpolate(self.final1(d1), size=target_size, mode="bilinear", align_corners=True)
                aux2 = F.interpolate(self.final2(d2), size=target_size, mode="bilinear", align_corners=True)
                aux3 = F.interpolate(self.final3(d3), size=target_size, mode="bilinear", align_corners=True)
                out_dict["aux"] = [aux1, aux2, aux3]

            return out_dict

        return output


def build_model(window_size=5, image_size=(256, 256), num_classes=1, input_channels=1):
    print(
        "[Model] Building Siamese STP-Net BDF-v1-1 with Shallow Reliability-Gated Skip, "
        f"Window Size: {window_size}, Input Size: {image_size}"
    )
    model = SiameseSTPromptNet(
        num_classes=num_classes,
        input_channels=input_channels,
        deep_supervision=True,
    )
    return model

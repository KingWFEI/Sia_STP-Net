import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
#  1. 基础组件
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
            nn.ReLU(inplace=True)
        )

    def forward(self, x): return self.conv(x)


class Up(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x): return self.up(x)


# =========================================================================
#  2. ASPP 与 孪生编码器
# =========================================================================
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1, bias=False), nn.BatchNorm2d(mid_channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 3, padding=6, dilation=6, bias=False),
                                   nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 3, padding=12, dilation=12, bias=False),
                                   nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 3, padding=18, dilation=18, bias=False),
                                   nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                                  nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(mid_channels * 5, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        x1, x2, x3, x4 = self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)
        x5 = F.interpolate(self.pool(x), size=x.shape[2:], mode='bilinear', align_corners=True)
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

class STNFeatureAligner(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(channels * 2, 32, 7, stride=2, padding=3), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(128 * 16, 128), nn.ReLU(True), nn.Linear(128, 6)
        )
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, feat_moving, feat_fixed):
        theta = self.localization(torch.cat([feat_moving, feat_fixed], dim=1)).view(-1, 2, 3)
        return F.grid_sample(feat_moving, F.affine_grid(theta, feat_moving.size(), align_corners=True),
                             align_corners=True)

class BiGatedDifferenceFusion(nn.Module):
    """
    轻量级双向差分门控时序融合模块
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.aligner = STNFeatureAligner(channels)

        # 第二步：构造方向性差分证据 (phi)
        in_ch = channels * 4 + 1  # [x_t, x_c, diff, abs_diff, PE]
        self.phi_evidence = nn.Sequential(
            nn.Conv2d(in_ch, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=channels),
            nn.ReLU(inplace=True)
        )

        # 第三步：生成可靠性门控 (psi) -> 输出 1 通道的空间可信度图
        self.psi_reliability = nn.Conv2d(channels, 1, kernel_size=1)

        # 第四步：轻量级状态更新门 (alpha)
        # 修正：输入为 [当前证据 e_t, 前一时刻状态 s_{t-1}]
        self.alpha_fwd = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.alpha_bwd = nn.Conv2d(channels * 2, channels, kernel_size=1)

        # 第五步：双向融合投影
        self.fusion_project = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            # 核心修改：将 BatchNorm2d 替换为不受小 Batch Size 影响的 GroupNorm
            nn.GroupNorm(num_groups=16, num_channels=channels),
            nn.ReLU(inplace=True)
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

        # -------------------------------------------------
        # Step 1: 所有帧对齐到目标帧坐标系
        # -------------------------------------------------
        aligned_seq = []
        for t in range(T):
            if t == target_idx:
                aligned_seq.append(feat_target)
            else:
                aligned_seq.append(self.aligner(feature_sequence[t], feat_target))

        # -------------------------------------------------
        # Step 2 & 3: 差分证据 + 可信度门控
        # -------------------------------------------------
        e_hat_seq = []
        gate_maps_rt = []

        for t in range(T):
            if t == target_idx:
                e_hat_seq.append(feat_target)
                gate_maps_rt.append(
                    torch.ones((B, 1, H, W), device=feat_target.device, dtype=feat_target.dtype)
                )
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
            e_hat_t = r_t * e_t

            e_hat_seq.append(e_hat_t)
            gate_maps_rt.append(r_t)

        # -------------------------------------------------
        # Step 4: 双向轻量状态累积
        # -------------------------------------------------
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

        # -------------------------------------------------
        # Step 5: 中心帧双向融合 + 残差锚定
        # -------------------------------------------------
        fwd_target_state = s_fwd_list[target_idx]
        bwd_target_state = s_bwd_list[target_idx]

        merged_state = self.fusion_project(torch.cat([fwd_target_state, bwd_target_state], dim=1))
        P_c = merged_state + feat_target

        return {
            "prompt": P_c,
            "gate_maps": torch.stack(gate_maps_rt, dim=1)   # [B, T, 1, H, W]，仅调试/可视化备用
        }
# =========================================================================
#  4. 时空动态提示词核心组件 (Temporal-Prompted Decoder Modules)
# =========================================================================
class TemporalPromptAttention(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, prompt, skip_feat):
        g1 = self.W_g(prompt)
        x1 = self.W_x(skip_feat)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)

        psi = self.relu(g1 + x1)
        attention_mask = self.psi(psi)
        activated_feat = skip_feat * attention_mask
        return activated_feat

class PromptDecoderBlock(nn.Module):
    def __init__(self, prompt_ch, skip_ch, out_ch):
        super().__init__()
        self.attention = TemporalPromptAttention(F_g=prompt_ch, F_l=skip_ch, F_int=skip_ch // 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(prompt_ch + skip_ch, out_ch)

    def forward(self, prompt, skip):
        activated_skip = self.attention(prompt, skip)
        prompt_up = self.up(prompt)
        return self.conv(torch.cat([prompt_up, activated_skip], dim=1))
# =========================================================================
#  5. 全新主模型: Siamese ST-Prompt-Net 带浅层可学习高频提取
# =========================================================================
class SiameseSTPromptNet(nn.Module):
    def __init__(self, num_classes, input_channels=1, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        nb_filter = [32, 64, 128, 256, 512]

        # 1. 孪生静态细节编码器
        self.encoder = SiameseEncoderWithASPP(input_channels, nb_filter)

        # 2. 浅层时序高频提取器
        self.temporal_extractor_0 = nn.Sequential(
            nn.Conv3d(in_channels=nb_filter[0], out_channels=nb_filter[0],
                      kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=nb_filter[0], bias=False),
            nn.BatchNorm3d(nb_filter[0]),
            nn.ReLU(inplace=True)
        )
        self.temporal_extractor_1 = nn.Sequential(
            nn.Conv3d(in_channels=nb_filter[1], out_channels=nb_filter[1],
                      kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=nb_filter[1], bias=False),
            nn.BatchNorm3d(nb_filter[1]),
            nn.ReLU(inplace=True)
        )

        # 3. 深层时空融合模块
        self.st_fusion_bottleneck = BiGatedDifferenceFusion(nb_filter[4])

        # 4. 解码器
        self.dec3 = PromptDecoderBlock(prompt_ch=nb_filter[4], skip_ch=nb_filter[3], out_ch=nb_filter[3])
        self.dec2 = PromptDecoderBlock(prompt_ch=nb_filter[3], skip_ch=nb_filter[2], out_ch=nb_filter[2])
        self.dec1 = PromptDecoderBlock(prompt_ch=nb_filter[2], skip_ch=nb_filter[1], out_ch=nb_filter[1])
        self.dec0 = PromptDecoderBlock(prompt_ch=nb_filter[1], skip_ch=nb_filter[0], out_ch=nb_filter[0])

        # 5. Deep Supervision 输出头
        self.final = nn.Conv2d(nb_filter[0], num_classes, 1)
        self.final1 = nn.Conv2d(nb_filter[1], num_classes, 1)
        self.final2 = nn.Conv2d(nb_filter[2], num_classes, 1)
        self.final3 = nn.Conv2d(nb_filter[3], num_classes, 1)

    def _extract_target_frame(self, feat_flat, B, T, target_idx):
        """纯静态提取：直接获取目标帧的空间细节"""
        _, C, H, W = feat_flat.shape
        return feat_flat.view(B, T, C, H, W)[:, target_idx, :, :, :]

    def _extract_sequence(self, feat_flat, B, T):
        """序列提取：供 ST_Fusion 的 LSTM 使用"""
        _, C, H, W = feat_flat.shape
        feat_seq_tensor = feat_flat.view(B, T, C, H, W)
        return [feat_seq_tensor[:, t, :, :, :] for t in range(T)]

    def _apply_temporal_filter(self, feat_flat, B, T, target_idx, filter_module):
        """
        动态特征提取：对展平的多帧特征应用 1D 可学习时间滤波器，
        由 Loss 反向传播指导网络抑制对齐噪声，增强真实微小破裂
        """
        _, C, H, W = feat_flat.shape
        # 1. 还原为 (B, T, C, H, W)
        feat_seq = feat_flat.view(B, T, C, H, W)
        # 2. 转换为 Conv3d 需要的维度排布 (B, C, T, H, W)
        feat_seq_3d = feat_seq.permute(0, 2, 1, 3, 4)
        # 3. 通过 Depthwise 1D 时间滤波
        filtered_seq_3d = filter_module(feat_seq_3d)
        # 4. 提取滤波后的目标帧 (此时该帧已融合了相邻帧的高频信息并被平滑了噪声)
        target_feat = filtered_seq_3d[:, :, target_idx, :, :]
        return target_feat

    def forward(self, x_sequence):
        B, T, C, H, W = x_sequence.shape
        target_idx = T // 2
        x_reshaped = x_sequence.view(B * T, C, H, W)

        # --- 第一阶段：编码静态细节 ---
        enc_feats_flat = self.encoder(x_reshaped)
        x0_flat, x1_flat, x2_flat, x3_flat, x4_flat = enc_feats_flat

        # -------------------------------------------------
        # 第二阶段：浅层时序增强
        # -------------------------------------------------
        x0_enriched = self._apply_temporal_filter(x0_flat, B, T, target_idx, self.temporal_extractor_0)
        x1_enriched = self._apply_temporal_filter(x1_flat, B, T, target_idx, self.temporal_extractor_1)

        x2_tgt = self._extract_target_frame(x2_flat, B, T, target_idx)
        x3_tgt = self._extract_target_frame(x3_flat, B, T, target_idx)

        # -------------------------------------------------
        # 第三阶段：深层时空融合
        # -------------------------------------------------
        x4_seq = self._extract_sequence(x4_flat, B, T)
        bottleneck_out = self.st_fusion_bottleneck(x4_seq)

        temporal_prompt_4 = bottleneck_out["prompt"]
        gate_maps = bottleneck_out["gate_maps"]  # 仅备用

        # -------------------------------------------------
        # 第四阶段：基于提示词的解码
        # -------------------------------------------------
        d3 = self.dec3(prompt=temporal_prompt_4, skip=x3_tgt)
        d2 = self.dec2(prompt=d3, skip=x2_tgt)
        d1 = self.dec1(prompt=d2, skip=x1_enriched)
        d0 = self.dec0(prompt=d1, skip=x0_enriched)

        output = self.final(d0)

        if self.training:
            out_dict = {
                "seg": output,
                "gate_maps": gate_maps  # 可留可不留，只做调试
            }

            if self.deep_supervision:
                target_size = output.shape[2:]
                aux1 = F.interpolate(self.final1(d1), size=target_size, mode='bilinear', align_corners=True)
                aux2 = F.interpolate(self.final2(d2), size=target_size, mode='bilinear', align_corners=True)
                aux3 = F.interpolate(self.final3(d3), size=target_size, mode='bilinear', align_corners=True)
                out_dict["aux"] = [aux1, aux2, aux3]

            return out_dict

        else:
            return output

#  6. 对外构建接口
# =========================================================================
def build_model(window_size=5, image_size=(256, 256), num_classes=1, input_channels=1):
    """
    基于时空动态提示词与浅层自适应高频滤波的全新网络 (Siamese ST-Prompt-Net v2)
    """
    print(f"[Model] Building Siamese ST-Prompt-Net with Window Size: {window_size}, Input Size: {image_size}")

    model = SiameseSTPromptNet(
        num_classes=num_classes,
        input_channels=input_channels,
        deep_supervision=True
    )

    return model

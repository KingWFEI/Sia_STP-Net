import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_2d_input(x, input_mode="stack"):
    """Adapt sequence input [B,T,C,H,W] to ordinary 2D segmentation input."""
    if x.dim() == 4:
        return x
    if x.dim() != 5:
        raise ValueError(f"Expected input shape [B,T,C,H,W] or [B,C,H,W], got {tuple(x.shape)}")

    b, t, c, h, w = x.shape
    if input_mode == "center":
        return x[:, t // 2]
    if input_mode == "stack":
        return x.reshape(b, t * c, h, w)
    raise ValueError(f"Unknown input_mode: {input_mode}")


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class Sequence2DWrapper(nn.Module):
    def __init__(self, model, input_mode="stack"):
        super().__init__()
        self.model = model
        self.input_mode = input_mode

    def forward(self, x):
        return self.model(_as_2d_input(x, self.input_mode))


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base_ch=32):
        super().__init__()
        f = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]
        self.pool = nn.MaxPool2d(2, 2)
        self.e1 = DoubleConv(in_channels, f[0])
        self.e2 = DoubleConv(f[0], f[1])
        self.e3 = DoubleConv(f[1], f[2])
        self.e4 = DoubleConv(f[2], f[3])
        self.center = DoubleConv(f[3], f[4])
        self.up4 = nn.ConvTranspose2d(f[4], f[3], 2, 2)
        self.d4 = DoubleConv(f[4], f[3])
        self.up3 = nn.ConvTranspose2d(f[3], f[2], 2, 2)
        self.d3 = DoubleConv(f[3], f[2])
        self.up2 = nn.ConvTranspose2d(f[2], f[1], 2, 2)
        self.d2 = DoubleConv(f[2], f[1])
        self.up1 = nn.ConvTranspose2d(f[1], f[0], 2, 2)
        self.d1 = DoubleConv(f[1], f[0])
        self.final = nn.Conv2d(f[0], num_classes, 1)

    @staticmethod
    def _cat(up, skip):
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([skip, up], dim=1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        center = self.center(self.pool(e4))
        d4 = self.d4(self._cat(self.up4(center), e4))
        d3 = self.d3(self._cat(self.up3(d4), e3))
        d2 = self.d2(self._cat(self.up2(d3), e2))
        d1 = self.d1(self._cat(self.up1(d2), e1))
        return self.final(d1)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base_ch=32):
        super().__init__()
        nb = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]
        self.pool = nn.MaxPool2d(2, 2)
        self.conv0_0 = DoubleConv(in_channels, nb[0])
        self.conv1_0 = DoubleConv(nb[0], nb[1])
        self.conv2_0 = DoubleConv(nb[1], nb[2])
        self.conv3_0 = DoubleConv(nb[2], nb[3])
        self.conv4_0 = DoubleConv(nb[3], nb[4])

        self.conv0_1 = DoubleConv(nb[0] + nb[1], nb[0])
        self.conv1_1 = DoubleConv(nb[1] + nb[2], nb[1])
        self.conv2_1 = DoubleConv(nb[2] + nb[3], nb[2])
        self.conv3_1 = DoubleConv(nb[3] + nb[4], nb[3])
        self.conv0_2 = DoubleConv(nb[0] * 2 + nb[1], nb[0])
        self.conv1_2 = DoubleConv(nb[1] * 2 + nb[2], nb[1])
        self.conv2_2 = DoubleConv(nb[2] * 2 + nb[3], nb[2])
        self.conv0_3 = DoubleConv(nb[0] * 3 + nb[1], nb[0])
        self.conv1_3 = DoubleConv(nb[1] * 3 + nb[2], nb[1])
        self.conv0_4 = DoubleConv(nb[0] * 4 + nb[1], nb[0])
        self.final = nn.Conv2d(nb[0], num_classes, 1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self._up(x1_0, x0_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up(x2_0, x1_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up(x1_1, x0_0)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up(x3_0, x2_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up(x2_1, x1_0)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up(x1_2, x0_0)], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up(x4_0, x3_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up(x3_1, x2_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up(x2_2, x1_0)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._up(x1_3, x0_0)], dim=1))
        return self.final(x0_4)


class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base_ch=32):
        super().__init__()
        self.stem = DoubleConv(in_channels, base_ch)
        self.low = DoubleConv(base_ch, base_ch * 2)
        self.mid = DoubleConv(base_ch * 2, base_ch * 4)
        self.high = DoubleConv(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2, 2)
        aspp_ch = base_ch * 8
        self.aspp1 = ConvBNReLU(aspp_ch, base_ch * 2, kernel_size=1, padding=0)
        self.aspp2 = ConvBNReLU(aspp_ch, base_ch * 2, kernel_size=3, padding=6)
        self.aspp2.block[0].dilation = (6, 6)
        self.aspp3 = ConvBNReLU(aspp_ch, base_ch * 2, kernel_size=3, padding=12)
        self.aspp3.block[0].dilation = (12, 12)
        self.aspp4 = ConvBNReLU(aspp_ch, base_ch * 2, kernel_size=3, padding=18)
        self.aspp4.block[0].dilation = (18, 18)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(aspp_ch, base_ch * 2, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.aspp_project = ConvBNReLU(base_ch * 10, base_ch * 4, kernel_size=1, padding=0)
        self.low_project = ConvBNReLU(base_ch * 2, base_ch, kernel_size=1, padding=0)
        self.decoder = nn.Sequential(
            DoubleConv(base_ch * 5, base_ch * 2),
            nn.Conv2d(base_ch * 2, num_classes, 1),
        )

    def forward(self, x):
        out_size = x.shape[-2:]
        x0 = self.stem(x)
        low = self.low(self.pool(x0))
        mid = self.mid(self.pool(low))
        high = self.high(self.pool(mid))
        pooled = F.interpolate(self.image_pool(high), size=high.shape[-2:], mode="bilinear", align_corners=False)
        aspp = torch.cat([self.aspp1(high), self.aspp2(high), self.aspp3(high), self.aspp4(high), pooled], dim=1)
        aspp = self.aspp_project(aspp)
        aspp = F.interpolate(aspp, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low = self.low_project(low)
        out = self.decoder(torch.cat([aspp, low], dim=1))
        return F.interpolate(out, size=out_size, mode="bilinear", align_corners=False)


class UNet3Plus(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base_ch=32):
        super().__init__()
        filters = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]
        self.pool = nn.MaxPool2d(2, 2)

        self.e1 = DoubleConv(in_channels, filters[0])
        self.e2 = DoubleConv(filters[0], filters[1])
        self.e3 = DoubleConv(filters[1], filters[2])
        self.e4 = DoubleConv(filters[2], filters[3])
        self.e5 = DoubleConv(filters[3], filters[4])

        cat_ch = base_ch
        up_ch = cat_ch * 5
        self.h1_to_hd4 = ConvBNReLU(filters[0], cat_ch)
        self.h2_to_hd4 = ConvBNReLU(filters[1], cat_ch)
        self.h3_to_hd4 = ConvBNReLU(filters[2], cat_ch)
        self.h4_to_hd4 = ConvBNReLU(filters[3], cat_ch)
        self.h5_to_hd4 = ConvBNReLU(filters[4], cat_ch)
        self.conv_hd4 = ConvBNReLU(up_ch, up_ch)

        self.h1_to_hd3 = ConvBNReLU(filters[0], cat_ch)
        self.h2_to_hd3 = ConvBNReLU(filters[1], cat_ch)
        self.h3_to_hd3 = ConvBNReLU(filters[2], cat_ch)
        self.hd4_to_hd3 = ConvBNReLU(up_ch, cat_ch)
        self.h5_to_hd3 = ConvBNReLU(filters[4], cat_ch)
        self.conv_hd3 = ConvBNReLU(up_ch, up_ch)

        self.h1_to_hd2 = ConvBNReLU(filters[0], cat_ch)
        self.h2_to_hd2 = ConvBNReLU(filters[1], cat_ch)
        self.hd3_to_hd2 = ConvBNReLU(up_ch, cat_ch)
        self.hd4_to_hd2 = ConvBNReLU(up_ch, cat_ch)
        self.h5_to_hd2 = ConvBNReLU(filters[4], cat_ch)
        self.conv_hd2 = ConvBNReLU(up_ch, up_ch)

        self.h1_to_hd1 = ConvBNReLU(filters[0], cat_ch)
        self.hd2_to_hd1 = ConvBNReLU(up_ch, cat_ch)
        self.hd3_to_hd1 = ConvBNReLU(up_ch, cat_ch)
        self.hd4_to_hd1 = ConvBNReLU(up_ch, cat_ch)
        self.h5_to_hd1 = ConvBNReLU(filters[4], cat_ch)
        self.conv_hd1 = ConvBNReLU(up_ch, up_ch)
        self.final = nn.Conv2d(up_ch, num_classes, 1)

    @staticmethod
    def _resize(x, size):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x):
        h1 = self.e1(x)
        h2 = self.e2(self.pool(h1))
        h3 = self.e3(self.pool(h2))
        h4 = self.e4(self.pool(h3))
        h5 = self.e5(self.pool(h4))

        hd4_size = h4.shape[-2:]
        hd4 = self.conv_hd4(torch.cat([
            self.h1_to_hd4(self._resize(h1, hd4_size)),
            self.h2_to_hd4(self._resize(h2, hd4_size)),
            self.h3_to_hd4(self._resize(h3, hd4_size)),
            self.h4_to_hd4(h4),
            self.h5_to_hd4(self._resize(h5, hd4_size)),
        ], dim=1))

        hd3_size = h3.shape[-2:]
        hd3 = self.conv_hd3(torch.cat([
            self.h1_to_hd3(self._resize(h1, hd3_size)),
            self.h2_to_hd3(self._resize(h2, hd3_size)),
            self.h3_to_hd3(h3),
            self.hd4_to_hd3(self._resize(hd4, hd3_size)),
            self.h5_to_hd3(self._resize(h5, hd3_size)),
        ], dim=1))

        hd2_size = h2.shape[-2:]
        hd2 = self.conv_hd2(torch.cat([
            self.h1_to_hd2(self._resize(h1, hd2_size)),
            self.h2_to_hd2(h2),
            self.hd3_to_hd2(self._resize(hd3, hd2_size)),
            self.hd4_to_hd2(self._resize(hd4, hd2_size)),
            self.h5_to_hd2(self._resize(h5, hd2_size)),
        ], dim=1))

        hd1_size = h1.shape[-2:]
        hd1 = self.conv_hd1(torch.cat([
            self.h1_to_hd1(h1),
            self.hd2_to_hd1(self._resize(hd2, hd1_size)),
            self.hd3_to_hd1(self._resize(hd3, hd1_size)),
            self.hd4_to_hd1(self._resize(hd4, hd1_size)),
            self.h5_to_hd1(self._resize(h5, hd1_size)),
        ], dim=1))
        return self.final(hd1)


class AttentionGate(nn.Module):
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.w_gate = nn.Sequential(nn.Conv2d(gate_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.w_skip = nn.Sequential(nn.Conv2d(skip_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, gate, skip):
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        attn = self.psi(F.relu(self.w_gate(gate) + self.w_skip(skip), inplace=True))
        return skip * attn


class UpAttentionBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.attn = AttentionGate(out_ch, skip_ch, max(out_ch // 2, 1))
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.attn(x, skip)
        return self.conv(torch.cat([skip, x], dim=1))


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base_ch=32):
        super().__init__()
        f = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]
        self.pool = nn.MaxPool2d(2, 2)
        self.e1 = DoubleConv(in_channels, f[0])
        self.e2 = DoubleConv(f[0], f[1])
        self.e3 = DoubleConv(f[1], f[2])
        self.e4 = DoubleConv(f[2], f[3])
        self.center = DoubleConv(f[3], f[4])
        self.d4 = UpAttentionBlock(f[4], f[3], f[3])
        self.d3 = UpAttentionBlock(f[3], f[2], f[2])
        self.d2 = UpAttentionBlock(f[2], f[1], f[1])
        self.d1 = UpAttentionBlock(f[1], f[0], f[0])
        self.final = nn.Conv2d(f[0], num_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        center = self.center(self.pool(e4))
        d4 = self.d4(center, e4)
        d3 = self.d3(d4, e3)
        d2 = self.d2(d3, e2)
        d1 = self.d1(d2, e1)
        return self.final(d1)


class TransUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=1,
        img_size=512,
        base_ch=32,
        transformer_dim=256,
        depth=4,
        num_heads=8,
        mlp_dim=512,
        dropout=0.0,
    ):
        super().__init__()
        f = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        self.e1 = DoubleConv(in_channels, f[0])
        self.e2 = DoubleConv(f[0], f[1])
        self.e3 = DoubleConv(f[1], f[2])
        self.e4 = DoubleConv(f[2], f[3])
        self.pool = nn.MaxPool2d(2, 2)

        self.patch_embed = nn.Conv2d(f[3], transformer_dim, kernel_size=2, stride=2)
        grid = max(img_size // 16, 1)
        self.pos_embed = nn.Parameter(torch.zeros(1, grid * grid, transformer_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.proj = ConvBNReLU(transformer_dim, f[3])

        self.up4 = nn.ConvTranspose2d(f[3], f[3], 2, 2)
        self.dec4 = DoubleConv(f[3] + f[3], f[3])
        self.up3 = nn.ConvTranspose2d(f[3], f[2], 2, 2)
        self.dec3 = DoubleConv(f[2] + f[2], f[2])
        self.up2 = nn.ConvTranspose2d(f[2], f[1], 2, 2)
        self.dec2 = DoubleConv(f[1] + f[1], f[1])
        self.up1 = nn.ConvTranspose2d(f[1], f[0], 2, 2)
        self.dec1 = DoubleConv(f[0] + f[0], f[0])
        self.final = nn.Conv2d(f[0], num_classes, 1)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _pos_embedding(self, h, w):
        n, c = self.pos_embed.shape[1], self.pos_embed.shape[2]
        old = int(math.sqrt(n))
        pos = self.pos_embed.transpose(1, 2).reshape(1, c, old, old)
        pos = F.interpolate(pos, size=(h, w), mode="bilinear", align_corners=False)
        return pos.flatten(2).transpose(1, 2)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))

        z = self.patch_embed(self.pool(e4))
        b, c, h, w = z.shape
        tokens = z.flatten(2).transpose(1, 2) + self._pos_embedding(h, w)
        tokens = self.transformer(tokens)
        z = tokens.transpose(1, 2).reshape(b, c, h, w)
        z = self.proj(z)

        d4 = self.up4(z)
        d4 = F.interpolate(d4, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = F.interpolate(d3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.final(d1)


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, h, w, c = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        hp, wp = x.shape[1], x.shape[2]

        x = x.view(b, hp // ws, ws, wp // ws, ws, c).permute(0, 1, 3, 2, 4, 5)
        windows = x.reshape(-1, ws * ws, c)
        qkv = self.qkv(windows).reshape(windows.shape[0], ws * ws, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(windows.shape[0], ws * ws, c)
        out = self.proj(out)
        out = out.view(b, hp // ws, wp // ws, ws, ws, c).permute(0, 1, 3, 2, 4, 5)
        out = out.reshape(b, hp, wp, c)
        if pad_h or pad_w:
            out = out[:, :h, :w, :]
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        shortcut = x
        x = x.permute(0, 2, 3, 1)
        y = self.norm1(x)
        if self.shift_size > 0:
            y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        y = self.attn(y)
        if self.shift_size > 0:
            y = torch.roll(y, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x.permute(0, 3, 1, 2).contiguous()


class SwinStage(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=7):
        super().__init__()
        blocks = []
        for i in range(depth):
            shift = 0 if i % 2 == 0 else window_size // 2
            blocks.append(SwinBlock(dim, num_heads, window_size, shift))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class PatchMerging(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.reduction = ConvBNReLU(in_ch, out_ch, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.reduction(x)


class PatchExpand(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.expand = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.expand(x)


class SwinUnet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base_ch=32, window_size=7):
        super().__init__()
        c = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        self.patch_embed = ConvBNReLU(in_channels, c[0], kernel_size=4, stride=4, padding=0)
        self.s1 = SwinStage(c[0], depth=2, num_heads=2, window_size=window_size)
        self.merge1 = PatchMerging(c[0], c[1])
        self.s2 = SwinStage(c[1], depth=2, num_heads=4, window_size=window_size)
        self.merge2 = PatchMerging(c[1], c[2])
        self.s3 = SwinStage(c[2], depth=2, num_heads=8, window_size=window_size)
        self.merge3 = PatchMerging(c[2], c[3])
        self.s4 = SwinStage(c[3], depth=2, num_heads=8, window_size=window_size)

        self.up3 = PatchExpand(c[3], c[2])
        self.d3 = nn.Sequential(DoubleConv(c[2] + c[2], c[2]), SwinStage(c[2], 1, 8, window_size))
        self.up2 = PatchExpand(c[2], c[1])
        self.d2 = nn.Sequential(DoubleConv(c[1] + c[1], c[1]), SwinStage(c[1], 1, 4, window_size))
        self.up1 = PatchExpand(c[1], c[0])
        self.d1 = nn.Sequential(DoubleConv(c[0] + c[0], c[0]), SwinStage(c[0], 1, 2, window_size))
        self.final_up = nn.ConvTranspose2d(c[0], c[0], kernel_size=4, stride=4)
        self.final = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x):
        out_size = x.shape[-2:]
        x1 = self.s1(self.patch_embed(x))
        x2 = self.s2(self.merge1(x1))
        x3 = self.s3(self.merge2(x2))
        x4 = self.s4(self.merge3(x3))

        d3 = self.up3(x4)
        d3 = F.interpolate(d3, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([d3, x3], dim=1))
        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([d2, x2], dim=1))
        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([d1, x1], dim=1))
        out = self.final(self.final_up(d1))
        if out.shape[-2:] != out_size:
            out = F.interpolate(out, size=out_size, mode="bilinear", align_corners=False)
        return out


def build_comparison_model(
    model_name,
    window_size=5,
    image_size=(512, 512),
    num_classes=1,
    input_channels=1,
    input_mode="stack",
    base_ch=32,
):
    model_key = model_name.lower().replace("-", "_")
    effective_in = input_channels if input_mode == "center" else input_channels * window_size

    if model_key in {"unet", "u_net"}:
        model = UNet(effective_in, num_classes, base_ch)
    elif model_key in {"unetpp", "unet++", "unet_plus_plus", "nested_unet"}:
        model = UNetPlusPlus(effective_in, num_classes, base_ch)
    elif model_key in {"deeplabv3plus", "deeplabv3_plus", "deeplab_v3_plus"}:
        model = DeepLabV3Plus(effective_in, num_classes, base_ch)
    elif model_key in {"2_5d_unet", "unet_2_5d", "two5d_unet"}:
        model = UNet(effective_in, num_classes, base_ch)
    elif model_key in {"unet3plus", "unet_3plus", "unet3+"}:
        model = UNet3Plus(effective_in, num_classes, base_ch)
    elif model_key in {"attention_unet", "att_unet", "attentionunet"}:
        model = AttentionUNet(effective_in, num_classes, base_ch)
    elif model_key == "transunet":
        model = TransUNet(
            effective_in,
            num_classes,
            img_size=image_size[0],
            base_ch=base_ch,
            transformer_dim=base_ch * 8,
            num_heads=8,
            mlp_dim=base_ch * 16,
        )
    elif model_key in {"swin_unet", "swinunet"}:
        model = SwinUnet(effective_in, num_classes, base_ch)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return Sequence2DWrapper(model, input_mode=input_mode)


COMPARISON_MODELS = (
    "unet",
    "unetpp",
    "deeplabv3plus",
    "unet3plus",
    "attention_unet",
    "transunet",
    "swin_unet",
    "2_5d_unet",
)

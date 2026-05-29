import json
import math
import os
import random
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as TF


# =========================================================================
#  基础几何工具函数
# =========================================================================

def get_cornea_roi_from_json(json_path, img_width, img_height):
    """从 labelme json 获取 cornea_roi 的矩形框"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return 0, 0, img_width, img_height

    shapes = data.get("shapes", [])
    cx = cy = r = None

    for shape in shapes:
        if shape.get("label") in ["cornea_roi", "circle"] or shape.get("shape_type") == "circle":
            pts = shape.get("points", [])
            if len(pts) >= 2:
                (cx, cy) = pts[0]
                (px, py) = pts[1]
                r = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
                break

    # 兜底：如果没有找到 circle，尝试找 polygon 的外接圆
    if cx is None:
        for shape in shapes:
            if shape.get("label") == "cornea_roi" and shape.get("shape_type") == "polygon":
                pts = np.array(shape.get("points"))
                (cx, cy), r = cv2.minEnclosingCircle(pts)
                break

    if cx is None or r is None:
        x1, y1, x2, y2 = 0, 0, img_width, img_height
    else:
        x1 = int(round(cx - r))
        x2 = int(round(cx + r))
        y1 = int(round(cy - r))
        y2 = int(round(cy + r))

        x1 = max(0, min(x1, img_width - 1))
        x2 = max(0, min(x2, img_width))
        y1 = max(0, min(y1, img_height - 1))
        y2 = max(0, min(y2, img_height))

        if x2 <= x1: x1, x2 = 0, img_width
        if y2 <= y1: y1, y2 = 0, img_height

    return x1, y1, x2, y2


def get_cornea_circle_from_json(json_path):
    """从 labelme json 获取圆参数 (cx, cy, r)"""
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    shapes = data.get("shapes", [])
    # 优先找 Label 为 cornea_roi 的
    for shape in shapes:
        if shape.get("label") == "cornea_roi":
            pts = shape.get("points", [])
            if shape.get("shape_type") == "circle" and len(pts) >= 2:
                (cx, cy) = pts[0]
                (px, py) = pts[1]
                r = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
                return float(cx), float(cy), float(r)
            elif shape.get("shape_type") == "polygon":
                pts_np = np.array(pts, dtype=np.int32)
                (cx, cy), r = cv2.minEnclosingCircle(pts_np)
                return float(cx), float(cy), float(r)

    # 其次找任意 circle 类型的
    for shape in shapes:
        if shape.get("shape_type") == "circle":
            pts = shape.get("points", [])
            if len(pts) >= 2:
                (cx, cy) = pts[0]
                (px, py) = pts[1]
                r = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
                return float(cx), float(cy), float(r)

    return None


def resize_and_pad(image, target_size, is_mask=False):
    """缩放并填充到正方形"""
    if image.ndim == 2:
        h, w = image.shape
    else:
        h, w, _ = image.shape

    scale = float(target_size) / max(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    border_type = cv2.BORDER_CONSTANT  # Mask 用 0 填充
    if not is_mask:
        border_type = cv2.BORDER_REPLICATE  # 图像用边缘复制填充

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, border_type)
    return padded


def _clip_int(v, lo, hi): return int(max(lo, min(hi, v)))


def _safe_rect(x1, y1, x2, y2, w, h):
    x1 = _clip_int(x1, 0, w - 1)
    x2 = _clip_int(x2, 0, w)
    y1 = _clip_int(y1, 0, h - 1)
    y2 = _clip_int(y2, 0, h)
    if x2 - x1 < 2: x1, x2 = 0, w
    if y2 - y1 < 2: y1, y2 = 0, h
    return x1, y1, x2, y2


def _safe_circle(cx, cy, r, w, h, min_r=8.0):
    cx = float(np.clip(cx, 0, w - 1))
    cy = float(np.clip(cy, 0, h - 1))
    r = float(max(r, min_r))
    return cx, cy, r


# =========================================================================
#   孪生网络模式
# =========================================================================

import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch.nn.functional as F


# 假设这些是你 utils.py 里已有的工具函数
# from utils import get_cornea_circle_from_json, PolarTransform

class SequencePMTMDataset(Dataset):
    """
    适配 ST-UNet++ 的滑动窗口序列数据集。
    输入: T 帧图像序列
    输出: (T, C, H, W) 图像张量, (1, H, W) 目标帧 Mask
    """

    def __init__(self, root_dir, window_size=5, img_size=640, use_polar=False, is_train=True):
        """
        Args:
            window_size (int): 滑动窗口大小 (T)，默认 5
        """
        self.root_dir = root_dir
        self.window_size = window_size
        self.img_size = img_size
        self.use_polar = use_polar
        self.is_train = is_train
        self.samples = []

        # 1. 初始化 PolarTransform (复用原有逻辑)
        # 注意：PolarTransform 继承自 nn.Module，通常在 CPU 上运行没问题
        self.polar_img_transform = PolarTransform(
            radial_resolution=img_size, angular_resolution=img_size,
            log_scale=True, min_radius=8.0, sample_mode='bilinear', binarize_output=False
        )
        self.polar_mask_transform = PolarTransform(
            radial_resolution=img_size, angular_resolution=img_size,
            log_scale=True, min_radius=8.0, sample_mode='nearest', binarize_output=True, binarize_threshold=0.5
        )

        # 2. 扫描逻辑
        self._scan_dataset()

    def _scan_dataset(self):
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Root dir not found: {self.root_dir}")

        patient_folders = sorted(
            [f for f in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, f))])
        print(f"Scanning {len(patient_folders)} patients (Window={self.window_size}, Train={self.is_train})...")

        for patient in patient_folders:
            p_path = os.path.join(self.root_dir, patient)
            img_dir = os.path.join(p_path, "images")
            mask_dir = os.path.join(p_path, "mask")
            json_dir = os.path.join(p_path, "json")

            if not os.path.exists(img_dir) or not os.path.exists(mask_dir): continue

            # 获取该病人所有帧并排序
            all_frames = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            if not all_frames: continue

            # 遍历每一帧，将其视作 "目标帧 (Target Frame, t)"
            for i in range(len(all_frames)):
                target_frame_name = all_frames[i]
                base_name = os.path.splitext(target_frame_name)[0]

                # 只有当目标帧有 Mask 时，才制作一个样本
                mask_path = os.path.join(mask_dir, base_name + ".png")
                json_path = os.path.join(json_dir, base_name + ".json")

                if os.path.exists(mask_path):
                    # --- 构建滑动窗口 ---
                    # 逻辑：取 [t-(T-1), ..., t] 这 T 帧
                    # 如果索引 < 0，则复制第一帧 (Edge Padding)
                    # seq_paths = []
                    # for k in range(self.window_size):
                    #     # 当前需要的帧索引：i - (window_size - 1) + k
                    #     # 例如 T=5, 当前是第10帧: 读取 6, 7, 8, 9, 10
                    #     frame_idx = i - (self.window_size - 1) + k
                    #     if frame_idx < 0:
                    #         frame_idx = 0  # 边界填充：复制首帧
                    #
                    #     seq_paths.append(os.path.join(img_dir, all_frames[frame_idx]))

                    # +++ 修改后的滑动窗口 (目标是中间帧) +++
                    seq_paths = []
                    half_window = self.window_size // 2  # 例如 T=5, half=2
                    for k in range(self.window_size):
                        # 以目标帧 i 为中心，左右扩展 half_window
                        frame_idx = i - half_window + k
                        # 边界处理：超出开头补第一帧，超出结尾补最后一帧
                        frame_idx = max(0, min(frame_idx, len(all_frames) - 1))

                        seq_paths.append(os.path.join(img_dir, all_frames[frame_idx]))

                    self.samples.append({
                        "seq_paths": seq_paths,  # List[str] 长度为 T
                        "target_json": json_path,  # 只记录目标帧的 JSON
                        "target_mask": mask_path  # 只记录目标帧的 Mask
                    })

        print(f"  -> Generated {len(self.samples)} sequence samples.")

    # def _load_tensor(self, img_path):
    #     img = cv2.imread(img_path)
    #     if img is None:
    #         # 异常处理：返回黑图
    #         return torch.zeros((3, 100, 100), dtype=torch.float32)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    def _load_tensor(self, img_path):
        # 1. 直接以单通道灰度模式读取图像，极大降低 I/O 开销
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # 异常处理：返回单通道的黑图 (1, H, W)
            return torch.zeros((1, 100, 100), dtype=torch.float32)

        # 2. 转换为 Tensor 并增加通道维度：从 (H, W) 变成 (1, H, W)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0

        return img_tensor

    def _crop_roi(self, tensor, cx, cy, r, margin=5):
        """简化的裁剪逻辑：只做裁剪，不做 resize (resize 留给 polar 或 interpolate)"""
        _, h, w = tensor.shape
        x1 = int(max(0, cx - r - margin))
        y1 = int(max(0, cy - r - margin))
        x2 = int(min(w, cx + r + margin))
        y2 = int(min(h, cy + r + margin))

        # 裁剪
        cropped = tensor[:, y1:y2, x1:x2]

        # 计算新圆心（相对于裁剪后的图）
        new_cx = float(cx - x1)
        new_cy = float(cy - y1)

        return cropped, new_cx, new_cy

    def __getitem__(self, idx):
        s = self.samples[idx]

        # =========================================================
        # 1. 获取定位参数 (仅基于 Target Frame)
        # =========================================================
        # 关键点：我们要训练对齐网络，所以必须用目标帧的圆心去切所有帧。
        # 这样前几帧如果发生了眼球转动，切出来的内容就会偏离中心，网络才能学到“怎么移回来”。
        dummy_h, dummy_w = 2000, 2000  # 兜底尺寸
        cx, cy, r = get_cornea_circle_from_json(s['target_json']) or (dummy_w / 2, dummy_h / 2, dummy_h / 3)

        # 增强策略：对齐抖动 (Alignment Jitter) - 对整个序列统一抖动圆心
        if self.is_train and random.random() < 0.6:
            cx += random.randint(-5, 5)
            cy += random.randint(-5, 5)

        # =========================================================
        # 2. 读取并处理序列 (T frames)
        # =========================================================
        processed_frames = []

        # 预先生成光照增强的参数 (让序列内部有细微差别，增加 LSTM 鲁棒性)
        brightness_factors = [random.uniform(0.8, 1.2) if (self.is_train and random.random() < 0.5) else 1.0 for _ in
                              range(self.window_size)]

        for i, path in enumerate(s['seq_paths']):
            # Load
            raw_tensor = self._load_tensor(path)

            # Augment: 亮度 (单帧独立增强，模拟闪烁/光照变化)
            if brightness_factors[i] != 1.0:
                raw_tensor = TF.adjust_brightness(raw_tensor, brightness_factors[i])

            # Crop: 所有帧都用同一个 (cx, cy, r)
            crop_tensor, new_cx, new_cy = self._crop_roi(raw_tensor, cx, cy, r)

            # Transform: 极坐标 或 Resize
            if self.use_polar:
                # 构造 Polar 参数
                cen = torch.tensor([[new_cx, new_cy]], dtype=torch.float32)
                rad = torch.tensor([r], dtype=torch.float32)
                # unsqueeze(0) 伪造 batch 维度
                with torch.no_grad():
                    final_frame = self.polar_img_transform(crop_tensor.unsqueeze(0), cen, rad)[0]
            else:
                # 笛卡尔坐标：直接 Resize 到 640x640
                final_frame = F.interpolate(
                    crop_tensor.unsqueeze(0),
                    size=(self.img_size, self.img_size),
                    mode='bilinear', align_corners=False
                )[0]

            processed_frames.append(final_frame)

        # Stack: (T, C, H, W)
        seq_tensor = torch.stack(processed_frames, dim=0)

        # =========================================================
        # 3. 处理 Target Mask
        # =========================================================
        _, h_raw, w_raw = self._load_tensor(s['seq_paths'][-1]).shape  # 获取尺寸用于 load mask
        mask_tensor = self._load_mask_tensor(s['target_mask'], h_raw, w_raw)

        # Crop Mask (同上)
        mask_crop, _, _ = self._crop_roi(mask_tensor, cx, cy, r)

        if self.use_polar:
            cen = torch.tensor([[new_cx, new_cy]], dtype=torch.float32)
            rad = torch.tensor([r], dtype=torch.float32)
            mask_final = self.polar_mask_transform(mask_crop.unsqueeze(0), cen, rad)[0]
        else:
            mask_final = F.interpolate(
                mask_crop.unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode='nearest'
            )[0]

        # =========================================================
        # 4. 序列级的一致性增强 (Consistent Augmentation)
        # =========================================================
        # 极坐标下的 Roll (相当于旋转) 和 Flip 必须对序列和 Mask 同时进行
        if self.is_train and self.use_polar:
            # 随机参数生成
            do_roll = random.random() < 0.5
            roll_shift = random.randint(-self.img_size // 10, self.img_size // 10)

            do_flip = random.random() < 0.5

            if do_roll:
                # roll 在最后一维 (W, 对应角度 theta)
                seq_tensor = torch.roll(seq_tensor, shifts=roll_shift, dims=-1)
                mask_final = torch.roll(mask_final, shifts=roll_shift, dims=-1)

            if do_flip:
                # flip 在最后一维
                seq_tensor = torch.flip(seq_tensor, dims=[-1])
                mask_final = torch.flip(mask_final, dims=[-1])

        return seq_tensor, mask_final

    # 辅助函数需包含在类内或外部
    def _load_mask_tensor(self, mask_path, ref_h, ref_w):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None: return torch.zeros((1, ref_h, ref_w), dtype=torch.float32)
        return torch.from_numpy(mask).float().unsqueeze(0) / 255.0

    def __len__(self):
        return len(self.samples)

# =========================================================================
#  辅助打印与统计
# =========================================================================

def print_metric_explanations():
    print("\n===== 指标说明 =====")
    print("Loss  : 总损失，模型训练优化目标，越低越好。")
    print("Dice  : Dice 系数（DSC），衡量预测与真实的重叠程度，1 表示完全一致。")
    print("IoU   : 交并比，预测区域与真实区域的相交面积 / 并集面积，越高越好。")
    print("Recall: 召回率，真实缺损中被模型检测出的比例，越高越好（减少漏检）。")
    print("FPR   : 假阳性率，背景中被误预测为缺损的概率，越低越好（减少误检）。")
    print("HD95  : 95% Hausdorff 距离，衡量预测边界与真实边界的最坏偏差，越低越好。")
    print("ASSD  : 平均对称表面距离，预测与真实边界的平均距离误差，越低越好。")
    print("====================\n")


def compute_all_metrics(pred_logits, target_mask, isVal=False):
    """
    计算所有指标 (Dice, IoU, Recall, FPR, HD95, ASSD)
    pred_logits: [B, 1, H, W] (未 sigmoid)
    target_mask: [B, 1, H, W] (0/1)
    """
    pred_prob = torch.sigmoid(pred_logits)
    pred_bin = (pred_prob > 0.5).float()

    # 简单的 Dice / IoU 计算
    smooth = 1e-5
    intersection = (pred_bin * target_mask).sum()
    union = pred_bin.sum() + target_mask.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)

    iou_union = pred_bin.sum() + target_mask.sum() - intersection
    iou = (intersection + smooth) / (iou_union + smooth)

    # Recall & FPR
    # TP: pred=1 & target=1
    tp = (pred_bin * target_mask).sum()
    # FN: pred=0 & target=1
    fn = ((1 - pred_bin) * target_mask).sum()
    # FP: pred=1 & target=0
    fp = (pred_bin * (1 - target_mask)).sum()
    # TN: pred=0 & target=0
    tn = ((1 - pred_bin) * (1 - target_mask)).sum()

    recall = (tp + smooth) / (tp + fn + smooth)
    fpr = (fp + smooth) / (fp + tn + smooth)

    # HD95 & ASSD 计算比较慢，通常只在 Validation 且 batch 不大时算
    # 为了速度，这里先返回 0.0，如果需要真实计算需引入 medpy 或 monai
    hd95 = 0.0
    assd = 0.0

    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "recall": recall.item(),
        "fpr": fpr.item(),
        "hd95": hd95,
        "assd": assd
    }


# =========================================================================
#  Polar Transformations (Module)
# =========================================================================

class PolarTransform(nn.Module):
    """
    RGB/灰度 -> 极域 (rho, theta)
    输出尺寸: (B, C, R, T)
    """

    def __init__(self, radial_resolution=512, angular_resolution=704,
                 log_scale=True, min_radius=8.0, sample_mode='bilinear',
                 binarize_output=False,
                 binarize_threshold=0.5):
        super().__init__()
        self.R = int(radial_resolution)
        self.T = int(angular_resolution)
        self.log_scale = bool(log_scale)
        self.min_radius = float(min_radius)
        self.sample_mode = sample_mode
        self.binarize_output = bool(binarize_output)
        self.binarize_threshold = float(binarize_threshold)

        theta = torch.linspace(0, 2 * math.pi, self.T, dtype=torch.float32)
        self.register_buffer("theta", theta, persistent=False)

    @torch.no_grad()
    def _build_radius(self, max_r: float, device):
        if self.log_scale:
            rho = torch.linspace(math.log(self.min_radius),
                                 math.log(max_r), self.R, device=device)
            r = torch.exp(rho)
        else:
            r = torch.linspace(self.min_radius, max_r, self.R, device=device)
        return r

    def forward(self, x: torch.Tensor, center: torch.Tensor, max_radius: torch.Tensor):
        B, C, H, W = x.shape
        device = x.device
        outs = []
        theta = self.theta.to(device)

        for b in range(B):
            r = self._build_radius(float(max_radius[b]), device)
            r_grid, th_grid = torch.meshgrid(r, theta, indexing='ij')

            cx, cy = center[b, 0], center[b, 1]
            xs = cx + r_grid * torch.cos(th_grid)
            ys = cy + r_grid * torch.sin(th_grid)

            gx = (xs / (W - 1)) * 2 - 1
            gy = (ys / (H - 1)) * 2 - 1
            grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)

            yb = F.grid_sample(
                x[b:b + 1], grid,
                mode=self.sample_mode,
                padding_mode='zeros',
                align_corners=True
            )
            outs.append(yb)
        y = torch.cat(outs, dim=0)
        if self.binarize_output:
            y = (y > self.binarize_threshold).float()
        return y


class PolarInverse(nn.Module):
    """
    极域 (B,C,R,T) -> 笛卡尔 (B,C,H,W) 回投
    """

    def __init__(self, out_h: int, out_w: int, log_scale=True, min_radius=8.0):
        super().__init__()
        self.H = int(out_h)
        self.W = int(out_w)
        self.log_scale = bool(log_scale)
        self.min_radius = float(min_radius)

        yy, xx = torch.meshgrid(
            torch.linspace(0, self.H - 1, self.H, dtype=torch.float32),
            torch.linspace(0, self.W - 1, self.W, dtype=torch.float32),
            indexing='ij'
        )
        self.register_buffer("base_xx", xx, persistent=False)
        self.register_buffer("base_yy", yy, persistent=False)

    def forward(self, x_polar: torch.Tensor, center: torch.Tensor, max_radius: torch.Tensor):
        B, C, R, T = x_polar.shape
        device = x_polar.device
        outs = []

        xx = self.base_xx.to(device)
        yy = self.base_yy.to(device)

        for b in range(B):
            cx, cy = center[b, 0], center[b, 1]
            dx = xx - cx
            dy = yy - cy
            rb = torch.sqrt(dx * dx + dy * dy)
            thetab = torch.atan2(dy, dx)
            thetab = (thetab + 2 * math.pi) % (2 * math.pi)

            if self.log_scale:
                rb_safe = torch.clamp(rb, min=self.min_radius)
                rho = (torch.log(rb_safe) - math.log(self.min_radius)) / \
                      (math.log(float(max_radius[b])) - math.log(self.min_radius) + 1e-8)
            else:
                rho = (rb - self.min_radius) / (float(max_radius[b]) - self.min_radius + 1e-8)

            rho = rho.clamp(0, 1)
            th = (thetab / (2 * math.pi)).clamp(0, 1 - 1e-6)

            gx = th * 2 - 1
            gy = rho * 2 - 1
            grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)

            yb = F.grid_sample(x_polar[b:b + 1], grid, mode='bilinear',
                               padding_mode='zeros', align_corners=True)
            outs.append(yb)

        return torch.cat(outs, dim=0)


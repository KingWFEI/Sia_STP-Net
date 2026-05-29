import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingBCE(nn.Module):
    """
    带标签平滑的二分类交叉熵损失 (Binary Cross Entropy with Label Smoothing)

    Args:
        pos_weight (Tensor or float, optional): 正样本权重，用于平衡类别不均。
        smoothing (float): 平滑系数 (alpha)。
                           例如 0.1 表示将标签从 {0, 1} 平滑到 {0.05, 0.95}。
    """

    def __init__(self, pos_weight=None, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

        # 确保 pos_weight 转换为 Tensor 并移动到设备上 (在 forward 中自动处理)
        if isinstance(pos_weight, (float, int)):
            self.register_buffer('pos_weight', torch.tensor([pos_weight]))
        else:
            self.register_buffer('pos_weight', pos_weight)

        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, logits, targets):
        """
        logits: 模型的原始输出 (未经过 Sigmoid), shape [B, 1, H, W]
        targets: 真实标签 (0 或 1), shape [B, 1, H, W]
        """
        # 1. 动态生成平滑标签
        # 公式: y_smooth = y * (1 - alpha) + 0.5 * alpha
        # 举例 (alpha=0.1):
        #   Target=1 -> 1 * 0.9 + 0.05 = 0.95
        #   Target=0 -> 0 * 0.9 + 0.05 = 0.05
        with torch.no_grad():
            smooth_targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing

        # 2. 计算 Loss
        loss = self.bce(logits, smooth_targets)
        return loss

class DiceLoss(nn.Module):
    """区域级 Dice 损失 (L_Dice)"""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, eps=1e-7):
        pred = pred.sigmoid()
        num = 2 * (pred * target).sum(dim=(2, 3))
        den = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 1 - (num + eps) / (den + eps)
        return dice.mean()


class MotionTolerantBoundaryLoss(nn.Module):
    """
    位移容忍的时空边界一致性损失
    输入应为已经处于同一参考坐标系下的时序预测序列
    shape: [B, T, 1, H, W]
    """

    def __init__(self, radius=3, beta=0.2, eps=1e-7):
        super().__init__()
        self.radius = radius
        self.beta = beta
        self.eps = eps

    def _soft_dilate(self, x):
        k = 2 * self.radius + 1
        return F.max_pool2d(x, kernel_size=k, stride=1, padding=self.radius)

    def _boundary_map(self, x):
        dx = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])
        dy = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])

        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        return dx + dy

    def _dice_loss(self, a, b):
        inter = (a * b).sum(dim=(2, 3, 4))
        union = a.sum(dim=(2, 3, 4)) + b.sum(dim=(2, 3, 4))
        dice = (2.0 * inter + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()

    def forward(self, pred_seq):
        """
        pred_seq: [B, T, 1, H, W]
        建议传入已经对齐到同一坐标系下的 logits 或 prob
        """
        probs = torch.sigmoid(pred_seq)

        prev = probs[:, :-1]   # [B, T-1, 1, H, W]
        curr = probs[:, 1:]    # [B, T-1, 1, H, W]

        # 1) 位移容忍的区域连续性
        prev_d = self._soft_dilate(prev.flatten(0, 1)).view_as(prev)
        curr_d = self._soft_dilate(curr.flatten(0, 1)).view_as(curr)
        loss_ov = self._dice_loss(prev_d, curr_d)

        # 2) 局部边界一致性
        b_prev = self._boundary_map(prev)
        b_curr = self._boundary_map(curr)
        band = self._soft_dilate(torch.maximum(prev, curr).flatten(0, 1)).view_as(prev).detach()
        loss_bd = (band * torch.abs(b_prev - b_curr)).mean()

        return loss_ov + self.beta * loss_bd


class TBUTGlobalLoss(nn.Module):
    """
    生理一致性时空协同混合损失 (Synergistic Spatiotemporal Loss)
    0.5 * L_lsbce (防误诊) + 0.5 * L_dice (精雕边缘) + lambda_ctc * L_RTM (内部物理约束)
    """

    def __init__(self, pos_weight=2.0, smoothing=0.1, lambda_temp =0.05):
        super().__init__()
        # 如果传入的是标量，将其转为 Tensor
        if isinstance(pos_weight, (float, int)):
            pos_weight = torch.tensor([float(pos_weight)])

        self.bce = LabelSmoothingBCE(pos_weight=pos_weight, smoothing=smoothing)
        self.dice = DiceLoss()
        self.temporal = MotionTolerantBoundaryLoss(radius=3, beta=0.2)
        self.lambda_temp = lambda_temp

    def _calc_main_loss(self, pred, target):
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
        return 0.5 * self.bce(pred, target) + 0.5 * self.dice(pred, target)

    def forward(self, outputs, targets):
        if not isinstance(outputs, dict):
            return self._calc_main_loss(outputs, targets)

        loss_main = self._calc_main_loss(outputs['seg'], targets)

        loss_aux = 0.0
        if 'aux' in outputs:
            aux_weights = [0.4, 0.2, 0.1]
            for i, aux_pred in enumerate(outputs['aux']):
                weight = aux_weights[i] if i < len(aux_weights) else 0.1
                loss_aux += weight * self._calc_main_loss(aux_pred, targets)

        loss_temp = 0.0
        if 'temporal_seq' in outputs and self.lambda_temp > 0:
            loss_temp = self.lambda_temp * self.temporal(outputs['temporal_seq'])

        return loss_main + loss_aux + loss_temp

class BaselineLossWrapper(nn.Module):
    """
    用来包装 LabelSmoothingBCE 和 DiceLoss 的基线损失函数。
    作为公平对比实验的统一损失标准。
    """

    def __init__(self, pos_weight=2.0, smoothing=0.1):
        super().__init__()
        self.criterion_bce = LabelSmoothingBCE(pos_weight=pos_weight, smoothing=smoothing)
        self.criterion_dice = DiceLoss()

    def _calc_single(self, pred, target):
        # 统一处理 size 不匹配的问题
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)

        l_bce = self.criterion_bce(pred, target)
        l_dice = self.criterion_dice(pred, target)
        return 0.5 * l_bce + 0.5 * l_dice

    def forward(self, outputs, targets):
        """
        兼容第三方模型输出 (Tensor) 和我们自定义模型输出 (Dict)
        """
        # 1. 如果是基线模型 (通常只返回一个 Tensor)
        if not isinstance(outputs, dict):
            return self._calc_single(outputs, targets)

        # 2. 如果是我们的模型 (返回 Dict，包含深层监督)
        loss = self._calc_single(outputs['seg'], targets)

        if 'aux' in outputs:
            aux_weights = [0.4, 0.2, 0.1]
            for i, aux_pred in enumerate(outputs['aux']):
                weight = aux_weights[i] if i < len(aux_weights) else 0.1
                loss += weight * self._calc_single(aux_pred, targets)

        return loss
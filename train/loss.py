import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class EnhancedSegmentationLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=2.0, boundary_weight=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.boundary_weight = boundary_weight
        self.smooth = 1e-5

    def focal_loss(self, input, target):
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = ((1 - pt) ** self.gamma * BCE_loss).mean()
        return focal_loss

    def tversky_loss(self, input, target):
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)


        input = torch.clamp(input, min=1e-7, max=1.0 - 1e-7)

        TP = (input * target).sum(1)
        FP = (input * (1 - target)).sum(1)
        FN = ((1 - input) * target).sum(1)


        Tversky = (TP + self.smooth) / (TP + self.beta * FP + (1 - self.beta) * FN + self.smooth)
        return 1 - torch.clamp(Tversky.mean(), min=0.0, max=1.0)

    def boundary_loss(self, input, target):
        input = torch.sigmoid(input)

        if input.dim() == 5:
            B, C, D, H, W = input.size()
            input = input.transpose(1, 2).contiguous().view(B * D, C, H, W)
            target = target.transpose(1, 2).contiguous().view(B * D, C, H, W)


        input = torch.clamp(input, min=1e-7, max=1.0 - 1e-7)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).cuda()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).cuda()
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)

        input_edges_x = F.conv2d(input, sobel_x, padding=1)
        input_edges_y = F.conv2d(input, sobel_y, padding=1)
        target_edges_x = F.conv2d(target, sobel_x, padding=1)
        target_edges_y = F.conv2d(target, sobel_y, padding=1)


        input_edges = torch.sqrt(torch.clamp(input_edges_x ** 2 + input_edges_y ** 2, min=1e-8))
        target_edges = torch.sqrt(torch.clamp(target_edges_x ** 2 + target_edges_y ** 2, min=1e-8))

        return F.mse_loss(input_edges, target_edges)

    def forward(self, input, target):
        focal = self.focal_loss(input, target)
        tversky = self.tversky_loss(input, target)
        boundary = self.boundary_loss(input, target)


        total_loss = torch.clamp(
            self.alpha * focal + (1 - self.alpha) * tversky + self.boundary_weight * boundary,
            min=1e-7,
            max=1e3
        )

        return total_loss
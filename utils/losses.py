import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from typing import Optional


def kl_divergence(p, q, eps=1e-8):
    p = p + eps  # Avoid log(0)
    q = q + eps  # Avoid log(0)
    return torch.sum(p * torch.log(p / q), dim=1)  # Sum over the distribution of each sample


def js_divergence(p, q, eps=1e-8):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)


class FuzzyClassificationLoss(nn.Module):
    """Fuzzy classification loss"""

    def __init__(self, weight: Optional[Tensor] = None, reduction: str = 'mean', softmax: bool = False):
        super(FuzzyClassificationLoss, self).__init__()
        if weight is not None and weight.min() < 0:
            raise ValueError('"weight" should be greater than or equal to 0.')
        self.weight = weight.unsqueeze(-1).unsqueeze(-1) if weight is not None else weight
        self.reduction = reduction
        self.softmax = softmax

    def forward(self, input: Tensor) -> Tensor:
        # if self.softmax:
        #     input = F.softmax(input, dim=1)
        # input = self.get_fuzzy_region(input)
        input = F.softmax(input, dim=1)
        if self.weight is None:
            # pixel_loss = -(1 - torch.prod(1 - input, dim=1)).log()
            pixel_loss = input[:, 0, :, :] * input[:, 1, :, :]
        else:
            pixel_loss = -(1 - torch.prod((1 - input).pow(self.weight), dim=1)).log()
        if self.reduction == 'mean':
            loss = pixel_loss.mean(dim=(0, 1, 2))
        elif self.reduction == 'sum':
            loss = pixel_loss.mean(dim=(1, 2)).sum(dim=0)
        else:
            loss = pixel_loss.mean(dim=(1, 2))
        return loss

    def get_fuzzy_region(self, input):
        # softmax
        input_prob = F.softmax(input, dim=1)
        # Pad the matrix with 1 unit for extracting 3x3 regions
        padded_output = F.pad(input_prob, pad=(1, 1, 1, 1), mode='constant', value=0)

        batch_size = len(input)
        fuzzy_regions = torch.empty(batch_size, 2, 3, 3)

        for i in range(batch_size):
            class1_probabilities = input[i, 1, :, :]
            _, max_index = torch.max(class1_probabilities.view(-1), 0)
            max_y, max_x = max_index // 7, max_index % 7
            # Adjust the index of the maximum position due to padding
            max_y_padded, max_x_padded = max_y + 1, max_x + 1
            # Extract a 3x3 region centered around the maximum probability position
            region = padded_output[i, :, max_y_padded-1:max_y_padded+2, max_x_padded-1:max_x_padded+2]
            fuzzy_regions[i] = region

        return fuzzy_regions


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, output, target):
        num_classes = output.size(1)
        assert len(self.alpha) == num_classes, \
            'Length of weight tensor must match the number of classes'
        logp = F.cross_entropy(output, target, self.alpha)
        p = torch.exp(-logp)
        focal_loss = (1-p)**self.gamma*logp

        return torch.mean(focal_loss)


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        """
        max_m: The appropriate value for max_m depends on the specific dataset and the severity of the class imbalance. 
        You can start with a small value and gradually increase it to observe the impact on the model's performance. 
        If the model struggles with class separation or experiences underfitting, increasing max_m might help. However,
        be cautious not to set it too high, as it can cause overfitting or make the model too conservative.

        s: The choice of s depends on the desired scale of the logits and the specific requirements of your problem. 
        It can be used to adjust the balance between the margin and the original logits. A larger s value amplifies 
        the impact of the logits and can be useful when dealing with highly imbalanced datasets. 
        You can experiment with different values of s to find the one that works best for your dataset and model.
        """
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index.bool(), x_m, x)  # fix
        return F.cross_entropy(self.s*output, target, weight=self.weight)


class LMFLoss(nn.Module):
    def __init__(self, cls_num_list, weight, alpha=1, beta=1, gamma=2, max_m=0.5, s=30):
        super().__init__()
        self.focal_loss = FocalLoss(weight, gamma)
        self.ldam_loss = LDAMLoss(cls_num_list, max_m, weight, s)
        self.alpha = alpha
        self.beta = beta

    def forward(self, output, target):
        focal_loss_output = self.focal_loss(output, target)
        ldam_loss_output = self.ldam_loss(output, target)
        total_loss = self.alpha*focal_loss_output + self.beta*ldam_loss_output
        return total_loss


class DIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        loss = torchvision.ops.distance_box_iou_loss(output, target, reduction='mean')
        return loss


class CIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        loss = torchvision.ops.complete_box_iou_loss(output, target, reduction='mean')
        return loss

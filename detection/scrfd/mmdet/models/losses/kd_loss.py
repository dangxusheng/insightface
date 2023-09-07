# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss


@weighted_loss
def knowledge_distillation_kl_div_loss(pred,
                                       soft_label,
                                       T,
                                       detach_target=True):
    r"""Loss function for knowledge distilling using KL divergence.

    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == soft_label.size()
    target = F.softmax(soft_label / T, dim=1)
    if detach_target:
        target = target.detach()

    kd_loss = F.kl_div(
        F.log_softmax(pred / T, dim=1), target, reduction='none').mean(1) * (
                      T * T)

    return kd_loss


@LOSSES.register_module()
class KnowledgeDistillationKLDivLoss(nn.Module):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, T=10):
        super(KnowledgeDistillationKLDivLoss, self).__init__()
        assert T >= 1
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T

    def forward(self,
                pred,
                soft_label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_kd = self.loss_weight * knowledge_distillation_kl_div_loss(
            pred,
            soft_label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            T=self.T)

        return loss_kd


@weighted_loss
def knowledge_distillation_smoothL1_loss(teacher_feat, stu_feat, beta=2.0):
    """
    使用特征图进行蒸馏, 为了保持特征图维度一样，需要在student得特征上增加1x1conv，作为projector
    Args:
        teacher_feat(Tensor):   [B,N,H,W]
        stu_feat(Tensor):       [B,N,H,W]

    Returns:
        torch.Tensor: Loss tensor with shape (B,).
    """
    assert teacher_feat.shape == stu_feat.shape, f'teacher_feat.shape != stu_feat.shape'
    # https://zhuanlan.zhihu.com/p/539929152
    # 对teach_feat 进行whiten, 使用layerNorm 来实现

    import torch.nn.functional as FF
    b, c, h, w = teacher_feat.shape
    teacher_feat_norm = FF.layer_norm(teacher_feat, normalized_shape=(c, h, w))
    # stu_feat_norm = FF.layer_norm(stu_feat, normalized_shape=(c,h,w))

    # print('teach.max, min:  ', teacher_feat_norm.max(), teacher_feat_norm.min())
    # print('stu.max, min:  ', stu_feat.max(), stu_feat.min())

    diff = (teacher_feat_norm.detach() - stu_feat).abs()
    import torch
    kd_loss = torch.zeros_like(diff)
    for i in range(b):
        for j in range(c):
            mask = diff[i, j] <= beta
            kd_loss[i, j][mask] = 0.5 * diff[i, j][mask].pow(2) / beta
            kd_loss[i, j][~mask] = diff[i, j][~mask] - beta / 2.

    kd_loss = kd_loss.sum((1, 2, 3))
    return kd_loss


@weighted_loss
def knowledge_distillation_L2_loss(teacher_feat, stu_feat):
    """
    使用特征图进行蒸馏, 为了保持特征图维度一样，需要在student得特征上增加1x1conv，作为projector
    Args:
        teacher_feat(Tensor):   [B,N,H,W]
        stu_feat(Tensor):       [B,N,H,W]

    Returns:
        torch.Tensor: Loss tensor with shape (B,).
    """
    assert teacher_feat.shape == stu_feat.shape, f'teacher_feat.shape != stu_feat.shape'
    # https://zhuanlan.zhihu.com/p/539929152
    # 对teach_feat 进行whiten, 使用layerNorm 来实现

    import torch.nn.functional as FF
    b, c, h, w = teacher_feat.shape
    teacher_feat_norm = FF.layer_norm(teacher_feat, normalized_shape=(c, h, w))
    # stu_feat_norm = FF.layer_norm(stu_feat, normalized_shape=(c,h,w))

    kd_loss = 0.5 * (teacher_feat_norm.detach() - stu_feat).pow(2).sqrt()
    kd_loss = kd_loss.sum((1, 2, 3))
    return kd_loss


@LOSSES.register_module()
class KnowledgeDistillationKLFeatLevelLoss(nn.Module):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self, is_smoothL1=True, reduction='mean', loss_weight=1.0, beta=2.0):
        super(KnowledgeDistillationKLFeatLevelLoss, self).__init__()
        assert beta > 0
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.beta = beta

        self.is_smoothL1 = is_smoothL1

    def forward(self,
                teacher_feat,
                stu_feat,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.is_smoothL1:
            loss_kd = self.loss_weight * knowledge_distillation_smoothL1_loss(
                teacher_feat,
                stu_feat,
                weight,
                reduction=reduction,
                avg_factor=avg_factor,
                beta=self.beta,
            )
        else:
            loss_kd = self.loss_weight * knowledge_distillation_L2_loss(
                teacher_feat,
                stu_feat,
                weight,
                reduction=reduction,
                avg_factor=avg_factor,
            )

        return loss_kd

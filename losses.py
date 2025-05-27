import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss']


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

    #
    # def forward(self, input, target):
    #
    #     bce_loss = F.binary_cross_entropy_with_logits(input, target)
    #
    #     # 为每个类别定义阈值
    #     threshold_1 = 0.0
    #     threshold_2 = 0.2
    #     threshold_3 = 0.4
    #     threshold_4 = 0.6
    #     threshold_5 = 0.8
    #
    #     # 使用阈值将预测分成六个类别
    #     class_0 = (target <= threshold_1)
    #     class_1 = ((target > threshold_1) & (target <= threshold_2))
    #     class_2 = ((target > threshold_2) & (target <= threshold_3))
    #     class_3 = ((target > threshold_3) & (target <= threshold_4))
    #     class_4 = ((target > threshold_4) & (target <= threshold_5))
    #     class_5 = (target >= threshold_5)
    #     num = target.size(0)
    #
    #     # 计算每个类别的 Dice Loss
    #     smooth = 1e-5
    #     #dice_0 = (2. * (input[class_0] * target[class_0]).sum() + smooth) / ( input[class_0].sum() + target[class_0].sum() + smooth)
    #     dice_1 = (2. * (input[class_1] * target[class_1]).sum() + smooth) / ( input[class_1].sum() + target[class_1].sum() + smooth)
    #     dice_2 = (2. * (input[class_2] * target[class_2]).sum() + smooth) / ( input[class_2].sum() + target[class_2].sum() + smooth)
    #     dice_3 = (2. * (input[class_3] * target[class_3]).sum() + smooth) / ( input[class_3].sum() + target[class_3].sum() + smooth)
    #     dice_4 = (2. * (input[class_4] * target[class_4]).sum() + smooth) / ( input[class_4].sum() + target[class_4].sum() + smooth)
    #     dice_5 = (2. * (input[class_5] * target[class_5]).sum() + smooth) / ( input[class_5].sum() + target[class_5].sum() + smooth)
    #
    #     dice1 = 1 - dice_1 / num
    #     dice2 = 1 - dice_2 / num
    #     dice3 = 1 - dice_3 / num
    #     dice4 = 1 - dice_4 / num
    #     dice5 = 1 - dice_5 / num
    #
    #     loss1 = 0.5 * bce_loss + dice1
    #     loss2 = 0.5 * bce_loss + dice2
    #     loss3 = 0.5 * bce_loss + dice3
    #     loss4 = 0.5 * bce_loss + dice4
    #     loss5 = 0.5 * bce_loss + dice5
    #
    #     # 返回损失函数值最大的那个类别的损失值
    #     loss = (loss1+loss2+loss3+loss4+loss5) / 5
    #     # max_loss = max( loss1, loss2, loss3, loss4, loss5)
    #     return loss










        # bce = F.binary_cross_entropy_with_logits(input, target)
        # smooth = 1e-5
        # input = torch.sigmoid(input)
        # num = target.size(0)
        # types_of_diseases = input.size(1)  # 假设有六个类别
        # input = input.view(num, types_of_diseases, -1)  # 每个类别的预测概率
        # target = target.view(num, types_of_diseases, -1)  # 每个类别的目标概率
        # intersection = (input * target)
        # dice = (2. * intersection.sum(2) + smooth) / (input.sum(2) + target.sum(2) + smooth)
        # dice = 1 - dice.sum(1) / types_of_diseases  # 每个样本的平均Dice分数
        # return 0.5 * bce + dice.mean()


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

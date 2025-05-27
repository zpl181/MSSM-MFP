import numpy as np
import torch
import torch.nn.functional as F


# def iou_score(output, target):
#     smooth = 1e-5
#     iou_scores = []
#     dice_scores = []
#
#     if torch.is_tensor(output):
#         output = torch.sigmoid(output).data.cpu().numpy()
#     if torch.is_tensor(target):
#         target = target.data.cpu().numpy()
#
#     for cls in range(5):
#         output_ = output > (cls + 1) * 0.2
#         target_ = target == (cls + 1) * 0.2
#         intersection = (output_ & target_).sum()
#         union = (output_ | target_).sum()
#         iou = (intersection + smooth) / (union + smooth)
#         dice = (2 * iou) / (iou + 1)
#         iou_scores.append(iou)
#         dice_scores.append(dice)
#     return max(iou_scores), max(dice_scores)


def iou_score(output, target):
    smooth = 1e-5
    # thresholds = [0.2, 0.4, 0.6, 0.8, 1.0]
    #
    # if torch.is_tensor(output):
    #     output = torch.sigmoid(output).data.cpu().numpy()
    # if torch.is_tensor(target):
    #     target = target.data.cpu().numpy()
    #
    # iou_scores = []
    # dice_scores = []
    #
    # for threshold in thresholds:
    #     output_ = output > 0.5
    #     target_ = target == threshold
    #
    #     intersection = (output_ & target_).sum()
    #     union = (output_ | target_).sum()
    #     iou = (intersection + smooth) / (union + smooth)
    #     dice = (2 * iou) / (iou + 1)
    #
    #     iou_scores.append(iou)
    #     dice_scores.append(dice)
    #
    # iou = sum(iou_scores) / len(iou_scores)
    # dice = sum(dice_scores) / len(dice_scores)

    if torch.is_tensor(output):
        #output = output.data.cpu().numpy()
        output = torch.sigmoid(output).data.cpu().numpy()
        #output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou+1)
    return iou, dice


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

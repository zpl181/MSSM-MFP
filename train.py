import argparse
import os
from collections import OrderedDict
from glob import glob

import numpy as np
import torch.nn.functional as F
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
import albumentations as albu
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90, Resize

import CTrans
import archs
import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool
import matplotlib.pyplot as plt
from archs import MSSM-MFP
import seaborn as sns

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=2, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='MSSM-MFP')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--types_of_diseases', default=6, type=int,
                        help='types of diseases')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: BCEDiceLoss)')

    # dataset
    parser.add_argument('--dataset', default='BUSI',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=60, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )

    parser.add_argument('--num_workers', default=1, type=int)

    config = parser.parse_args()

    return config

# args = parser.parse_args()
def calculate_pre_recall_f1_acc(output, target):
    # 这里添加计算 SE、PC、F1、ACC 的逻辑
    # 示例：计算混淆矩阵
    pred_binary = (output > 0.5).float()  # 假设输出为概率，二值化为0和1
    true_positive = torch.sum(pred_binary * target)
    false_positive = torch.sum(pred_binary * (1 - target))
    false_negative = torch.sum((1 - pred_binary) * target)
    true_negative = torch.sum((1 - pred_binary) * (1 - target))

    sensitivity = true_positive / (true_positive + false_negative + 1e-10)
    specificity = true_negative / (true_negative + false_positive + 1e-10)
    precision = true_positive / (true_positive + false_positive + 1e-10)
    recall = sensitivity
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    accuracy = (true_positive + true_negative) / (
                true_positive + false_positive + false_negative + true_negative + 1e-10)

    return precision.item(), recall.item(), f1_score.item(), accuracy.item()

def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'Pre': AverageMeter(),
                  'Recall': AverageMeter(),
                  'F1': AverageMeter(),
                  'ACC': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou, dice = iou_score(outputs[-1], target)
            Pre, Recall, f1, acc = calculate_pre_recall_f1_acc(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou, dice = iou_score(output, target)
            Pre, Recall, f1, acc = calculate_pre_recall_f1_acc(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['Pre'].update(Pre, input.size(0))
        avg_meters['Recall'].update(Recall, input.size(0))
        avg_meters['F1'].update(f1, input.size(0))
        avg_meters['ACC'].update(acc, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('Pre', avg_meters['Pre'].avg),
            ('Recall', avg_meters['Recall'].avg),
            ('F1', avg_meters['F1'].avg),
            ('ACC', avg_meters['ACC'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg),
        ('Pre', avg_meters['Pre'].avg),
        ('Recall', avg_meters['Recall'].avg),
        ('F1', avg_meters['F1'].avg),
        ('ACC', avg_meters['ACC'].avg),
    ])

# 同样，在验证函数中添加相应的指标计算和记录
def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'Pre': AverageMeter(),
                  'Recall': AverageMeter(),
                  'F1': AverageMeter(),
                  'ACC': AverageMeter(),
                  'dice': AverageMeter()}

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, dice = iou_score(outputs[-1], target)
                Pre, Recall, f1, acc = calculate_pre_recall_f1_acc(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, dice = iou_score(output, target)
                Pre, Recall, f1, acc = calculate_pre_recall_f1_acc(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['Pre'].update(Pre, input.size(0))
            avg_meters['Recall'].update(Recall, input.size(0))
            avg_meters['F1'].update(f1, input.size(0))
            avg_meters['ACC'].update(acc, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('Pre', avg_meters['Pre'].avg),
                ('Recall', avg_meters['Recall'].avg),
                ('F1', avg_meters['F1'].avg),
                ('ACC', avg_meters['ACC'].avg),
                ('dice', avg_meters['dice'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg),
        ('Pre', avg_meters['Pre'].avg),
        ('Recall', avg_meters['Recall'].avg),
        ('F1', avg_meters['F1'].avg),
        ('ACC', avg_meters['ACC'].avg),
        ('dice', avg_meters['dice'].avg),
    ])


# def validate(config, val_loader, model, criterion):
#     avg_meters = {'loss': AverageMeter(),
#                   'iou': AverageMeter(),
#                   'dice': AverageMeter()}
#
#     # switch to evaluate mode
#     model.eval()
#
#     with torch.no_grad():
#         pbar = tqdm(total=len(val_loader))
#         for input, target, _ in val_loader:
#
#             input = input.cuda()
#             target = target.cuda()
#
#             # compute output
#             if config['deep_supervision']:
#                 outputs = model(input)
#                 loss = 0
#                 for output in outputs:
#                     loss += criterion(output, target)
#                 loss /= len(outputs)
#                 iou, dice = iou_score(outputs[-1], target)
#             else:
#                 output = model(input)
#                 loss = criterion(output, target)
#                 iou, dice = iou_score(output, target)
#
#             avg_meters['loss'].update(loss.item(), input.size(0))
#             avg_meters['iou'].update(iou, input.size(0))
#             avg_meters['dice'].update(dice, input.size(0))
#
#             postfix = OrderedDict([
#                 ('loss', avg_meters['loss'].avg),
#                 ('iou', avg_meters['iou'].avg),
#                 ('dice', avg_meters['dice'].avg)
#             ])
#             pbar.set_postfix(postfix)
#             pbar.update(1)
#         pbar.close()
#
#     return OrderedDict([('loss', avg_meters['loss'].avg),
#                         ('iou', avg_meters['iou'].avg),
#                         ('dice', avg_meters['dice'].avg)])


def validate_t(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'Pre': AverageMeter(),
                  'Recall': AverageMeter(),
                  'F1': AverageMeter(),
                  'ACC': AverageMeter(),
                  'dice': AverageMeter()}

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, dice = iou_score(outputs[-1], target)
                Pre, Recall, f1, acc = calculate_pre_recall_f1_acc(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, dice = iou_score(output, target)
                Pre, Recall, f1, acc = calculate_pre_recall_f1_acc(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['Pre'].update(Pre, input.size(0))
            avg_meters['Recall'].update(Recall, input.size(0))
            avg_meters['F1'].update(f1, input.size(0))
            avg_meters['ACC'].update(acc, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('Pre', avg_meters['Pre'].avg),
                ('Recall', avg_meters['Recall'].avg),
                ('F1', avg_meters['F1'].avg),
                ('ACC', avg_meters['ACC'].avg),
                ('dice', avg_meters['dice'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg),
        ('Pre', avg_meters['Pre'].avg),
        ('Recall', avg_meters['Recall'].avg),
        ('F1', avg_meters['F1'].avg),
        ('ACC', avg_meters['ACC'].avg),
        ('dice', avg_meters['dice'].avg),
    ])

# def validate_t(config, test_loader, model, criterion):
#     avg_meters = {'loss': AverageMeter(),
#                   'iou': AverageMeter(),
#                   'dice': AverageMeter()}
#
#     # switch to evaluate mode
#     model.eval()
#
#     with torch.no_grad():
#         pbar = tqdm(total=len(test_loader))
#         for input, target, _ in test_loader:
#
#             input = input.cuda()
#             target = target.cuda()
#
#             # compute output
#             if config['deep_supervision']:
#                 outputs = model(input)
#                 loss = 0
#                 for output in outputs:
#                     loss += criterion(output, target)
#                 loss /= len(outputs)
#                 iou, dice = iou_score(outputs[-1], target)
#             else:
#                 output = model(input)
#                 loss = criterion(output, target)
#                 iou, dice = iou_score(output, target)
#
#             avg_meters['loss'].update(loss.item(), input.size(0))
#             avg_meters['iou'].update(iou, input.size(0))
#             avg_meters['dice'].update(dice, input.size(0))
#
#             postfix = OrderedDict([
#                 ('loss', avg_meters['loss'].avg),
#                 ('iou', avg_meters['iou'].avg),
#                 ('dice', avg_meters['dice'].avg)
#             ])
#             pbar.set_postfix(postfix)
#             pbar.update(1)
#         pbar.close()
#
#     return OrderedDict([('loss', avg_meters['loss'].avg),
#                         ('iou', avg_meters['iou'].avg),
#                         ('dice', avg_meters['dice'].avg)])


def main():
    config = vars(parse_args())
    #os.makedirs('checkpoint/%s' % config['name'], exist_ok=True)

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])

    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)


    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

 #   # Data loading code
  #  img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
  #  img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

  #  total_samples = len(img_ids)
 #   train_ratio = 0.8
#    val_ratio = 0.1
 #   test_ratio = 0.1

 #   train_split = int(total_samples * train_ratio)
#    val_split = train_split + int(total_samples * val_ratio)

 #   train_img_ids = img_ids[:train_split]
  #  val_img_ids = img_ids[train_split:val_split]
   # test_img_ids = img_ids[val_split:]

    # Data loading code
    train_img_ids = glob(
        os.path.join('E:\\MSSM-MFP\\inputs\\BUSI\\train_images\\images', '*' + config['img_ext']))  # 指定训练集路径
    val_img_ids = glob(os.path.join('E:\\MSSM-MFP\\inputs\\BUSI\\val_images\\images', '*' + config['img_ext']))  # 指定验证集路径
    test_img_ids = glob(os.path.join('E:\\MSSM-MFP\\inputs\\BUSI\\test_images\\images', '*' + config['img_ext']))  # 指定测试集路径

    # 提取文件名（不带扩展名）
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]
    test_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_img_ids]


    train_transform = Compose([
        RandomRotate90(),
        albu.Flip(),
        Resize(height=config['input_h'], width=config['input_w']),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(height=config['input_h'], width=config['input_w']),
        transforms.Normalize(),
    ])

    test_transform = Compose([
        Resize(height=config['input_h'], width=config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('E:\\MSSM-MFP\\inputs\\BUSI\\train_images\\images'),  # 指定测试集路径
        mask_dir=os.path.join('E:\\MSSM-MFP\\inputs\\BUSI\\train_images\\masks'),  # 指定测试集掩码路径
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('E:\\MSSM-MFP\\inputs\\BUSI\\val_images\\images'),  # 指定测试集路径
        mask_dir=os.path.join('E:\\MSSM-MFP\\inputs\\BUSI\\val_images\\masks'),  # 指定测试集掩码路径
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    test_dataset = Dataset(
        img_ids=test_img_ids,
        img_dir=os.path.join('E:\\MSSM-MFP\\inputs\\BUSI\\test_images\\images'),  # 指定测试集路径
        mask_dir=os.path.join('E:\\MSSM-MFP\\inputs\\BUSI\\test_images\\masks'),  # 指定测试集掩码路径
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)

    # for item in train_loader:
    #     print(1)
    #     break
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    # create model

    model = archs.__dict__[config['arch']](1)

    model = model.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    # 学习率策略
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # 创建字典
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('Pre', []),
        ('Recall', []),
        ('F1', []),
        ('ACC', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
        ('val_Pre', []),
        ('val_Recall', []),
        ('val_F1', []),
        ('val_ACC', []),
        ('test_iou', []),
        ('test_dice', []),
        ('test_Pre', []),
        ('test_Recall', []),
        ('test_F1', []),
        ('test_ACC', []),
    ])

    best_iou = 0
    trigger = 0

    train_metrics = []
    val_metrics = []
    test_metrics = []

    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        test_log = validate_t(config, test_loader, model, criterion)

        train_metrics.append(
            [train_log['loss'], train_log['iou'], train_log['Pre'], train_log['Recall'], train_log['F1'],
             train_log['ACC']])
        val_metrics.append(
            [val_log['loss'], val_log['iou'], val_log['Pre'], val_log['Recall'], val_log['F1'], val_log['ACC']])
        test_metrics.append(
            [test_log['loss'], test_log['iou'], test_log['Pre'], test_log['Recall'], test_log['F1'], test_log['ACC']])

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print(
            'loss %.4f - iou %.4f - Pre %.4f - Recall %.4f - F1 %.4f - ACC %.4f - val_loss %.4f - val_iou %.4f - val_Pre %.4f - val_Recall %.4f - val_F1 %.4f - val_ACC %.4f - test_loss %.4f - test_iou %.4f - test_Pre %.4f - test_Recall %.4f - test_F1 %.4f - test_ACC %.4f'
            % (train_log['loss'], train_log['iou'], train_log['Pre'], train_log['Recall'], train_log['F1'], train_log['ACC'],
               val_log['loss'], val_log['iou'], val_log['Pre'], val_log['Recall'], val_log['F1'], val_log['ACC'],
               test_log['loss'], test_log['iou'], test_log['Pre'], test_log['Recall'], test_log['F1'], test_log['ACC']))

        print(f'Learning Rate: {scheduler.get_last_lr()[0]}')

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['Pre'].append(train_log['Pre'])
        log['Recall'].append(train_log['Recall'])
        log['F1'].append(train_log['F1'])
        log['ACC'].append(train_log['ACC'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['val_Pre'].append(val_log['Pre'])
        log['val_Recall'].append(val_log['Recall'])
        log['val_F1'].append(val_log['F1'])
        log['val_ACC'].append(val_log['ACC'])
        log['test_iou'].append(test_log['iou'])
        log['test_dice'].append(test_log['dice'])
        log['test_Pre'].append(test_log['Pre'])
        log['test_Recall'].append(test_log['Recall'])
        log['test_F1'].append(test_log['F1'])
        log['test_ACC'].append(test_log['ACC'])

        #pd.DataFrame(log).to_csv('checkpoint/%s/log.csv' % config['name'], index=False)
        pd.DataFrame(log).to_csv('models/%s/log.csv' % config['name'], index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            #torch.save(model.state_dict(), 'checkpoint/%s/model.pth' % config['name'])
            torch.save(model.state_dict(), 'models/%s/model.pth' % config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()



if __name__ == '__main__':
    main()

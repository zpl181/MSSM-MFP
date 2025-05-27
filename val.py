import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchsummary import summary
from thop import profile

import archs1
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90, Resize
import time
from archs import MSSM-MFP

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="BUSI_MSSM-MFP_woDS", help='model name')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs1.__dict__[config['arch']](1)
    model = model.cuda()

    # # Data loading code
    # img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    # img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    #
    # total_samples = len(img_ids)
    # train_ratio = 0.8  # 指定训练集所占的比例
    # val_ratio = 0.1  # 指定验证集所占的比例
    # test_ratio = 0.1  # 指定测试集所占的比例
    #
    # train_split = int(total_samples * train_ratio)
    # val_split = train_split + int(total_samples * val_ratio)
    #
    # train_img_ids = img_ids[:train_split]  # 前 train_split 个样本作为训练集
    # val_img_ids = img_ids[train_split:val_split]  # 接下来 val_split 个样本作为验证集
    # test_img_ids = img_ids[val_split:]  # 剩余的样本作为测试集
    #
    # print(train_img_ids)
    # print(val_img_ids)
    # print(test_img_ids)
    # train_img_ids, temp_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
    # _, val_img_ids = train_test_split(temp_img_ids, test_size=0.5, random_state=42)

    # Data loading code
    train_img_ids = glob(
        os.path.join('E:\\MSSM-MFP\\inputs\\BUSI\\train_images\\images', '*' + config['img_ext']))  # 指定训练集路径
    val_img_ids = glob(os.path.join('E:\\MSSM-MFP\\inputs\\BUSI\\val_images\\images', '*' + config['img_ext']))  # 指定验证集路径
    test_img_ids = glob(os.path.join('E:\\MSSM-MFP\\inputs\\BUSI\\test_images\\images', '*' + config['img_ext']))  # 指定测试集路径

    # 提取文件名（不带扩展名）
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]
    test_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_img_ids]

    model.load_state_dict(torch.load('models/%s/model.pth' % config['name']))
    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=test_img_ids,
        img_dir=os.path.join('E:\\MSSM-MFP\\inputs\\BUSI\\test_images\\images'),  # 指定测试集路径
        mask_dir=os.path.join('E:\\MSSM-MFP\\inputs\\BUSI\\test_images\\masks'),  # 指定测试集掩码路径
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    gput = AverageMeter()
    cput = AverageMeter()

    count = 0
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)

    # input_tensor = torch.randn(1, config['input_channels'], config['input_h'], config['input_w']).cuda()  # Create a dummy input
    # flops, params = profile(model, inputs=(input_tensor,))  # Calculate FLOPS and Params
    # print(f'FLOPS: {flops / 1e9:.2f} GFLOPS')  # Convert to GFLOPS
    # print(f'Params: {params / 1e6:.2f} M')  # Convert to millions

    # Start inference time measurement
    start_time = time.time()
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()   #图片
            target = target.cuda()  #mask标记
            model = model.cuda()
            # compute output
            output = model(input)
            #输出图片
            # Modify the output based on pixel valuesla to classify into six categories
            threshold_1 = 0.0  # Define thresholds for classification
            threshold_2 = 1.0
            threshold_3 = 2.0
            threshold_4 = 3.0
            threshold_5 = 4.0
            threshold_6 = 5.0



            iou, dice = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))
            # thresholds = [threshold_1, threshold_2, threshold_3, threshold_4, threshold_5, threshold_6]
            #
            # # Create an array for class labels
            # class_labels = [0, 1, 2, 3, 4, 5]

            # Initialize the output as all zeros
            #output = torch.zeros_like(output)

            # Apply thresholds to classify pixels into classes
            # for i in range(len(class_labels)):
            #     if i == len(class_labels) - 1:
            #         mask = (output > thresholds[i])
            #     else:
            #         mask = (output > thresholds[i]) & (output <= thresholds[i + 1])
            #     output[mask] = class_labels[i]
            output = torch.sigmoid(output)  # .cpu().numpy()

            # output[output <= threshold_1] = 0.0
            # output[(output > threshold_1) & (output <= threshold_2)] = 1.0
            # output[(output > threshold_2) & (output <= threshold_3)] = 2.0
            # output[(output > threshold_3) & (output <= threshold_4)] = 3.0
            # output[(output > threshold_4) & (output <= threshold_5)] = 4.0
            # output[output > threshold_5] = 5.0



            # Save the modified output
            for i in range(len(output)):
                for c in range(config['num_classes']):
                    output_image = ((output[i, c] * 255).cpu().numpy()).astype('uint8')
                    #output_image = (output[i, c].cpu().numpy() * 255.0).astype('uint8')
                    # output[output <= threshold_1] = 0
                    # output[(output > threshold_1) & (output <= threshold_2)] = 1
                    # output[(output > threshold_2) & (output <= threshold_3)] = 2
                    # output[(output > threshold_3) & (output <= threshold_4)] = 3
                    # output[(output > threshold_4) & (output <= threshold_5)] = 4
                    # output[(output > threshold_5) & (output <= threshold_6)] = 5
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'), output_image)

    inference_time = time.time() - start_time
    print(f'Inference Time: {inference_time:.4f} seconds')

    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)

    #print('CPU: %.4f' % cput.avg)
    # print('GPU: %.4f' % gput.avg)

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()

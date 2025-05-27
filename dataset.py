import os

import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None,):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.

        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        # img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext), cv2.IMREAD_GRAYSCALE)
        # 复制单通道图像到三通道
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #img = cv2.resize(img, (256, 256))



        #img = img.transpose(2, 0, 1)
        # img_resize = np.zeros((self.num_classes, img.shape[1], img.shape[2],), dtype=np.float32)
        # img_resize = np.repeat(img_resize, 3, axis=0)
        # for i in range(3):
        #     img_resize[i, :, :] = img[0, :, :]
        # mask = np.zeros((img.shape[0], img.shape[1], self.num_classes), dtype=np.float32)
        # mask_list = []
        mask = []

        for i in range(self.num_classes):
            # mask_path = os.path.join(self.mask_dir, str(i), img_id + self.mask_ext)
            # mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # if mask_img is not None:
            # 将图像转换为单通道
            mask_channel = cv2.imread(os.path.join(self.mask_dir, str(i), img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)
            #mask_channel = cv2.resize(mask_channel, (256, 256))



            # k.concate(mask_channel)
            mask.append(mask_channel)
        mask = np.dstack(mask)
        #mask = mask.transpose(2, 0, 1)

        # mask = mask.transpose(2, 0, 1) / 255.0
        # mask[..., i] = mask_img.astype(np.float32)
        # mask_list.append(mask_img)
        # else:
        #     mask[..., i] = 0.0
        #     mask_list.append(np.zeros_like(mask_img))

        # 扩展维度并添加到列表
        #         mask_img = mask_img[..., None]
        #         mask.append(mask_img)
        # mask = np.dstack(mask_list)
        # mask = np.dstack(mask)
        mask_resize = np.zeros((img.shape[0], img.shape[1],self.num_classes), dtype=np.float32)
        #mask_resize = np.repeat(mask_resize, 3, axis=0)
        for i in range(1):
            mask_resize[:, :, i] = mask[ :, :, 0]

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask_resize)
            img = augmented['image']
            mask_resize = augmented['mask']

        # threshold = 0.0
        # img[img <= threshold] = 0.0

        img = img.astype('float32')
        img = img.transpose(2, 0, 1) / 255.0
        mask_resize = mask_resize.astype('float32')
        mask_resize = mask_resize.transpose(2, 0, 1) / 255.0

        return img, mask_resize, {'img_id': img_id}

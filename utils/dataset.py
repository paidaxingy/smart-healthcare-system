import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import numpy as np
import random

class MedicalSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list, img_size=512, mode='train'):
        """
        通用医学图像分割数据集
        image_dir: 图像文件夹路径
        mask_dir:  Mask文件夹路径
        file_list: 图像文件名列表（不含扩展名）
        img_size:  统一尺寸，默认512x512
        mode:     'train', 'val', 'test'
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.file_list = file_list
        self.img_size = img_size
        self.mode = mode

    def __len__(self):
        return len(self.file_list)

    def random_crop(self, img, mask, crop_size):
        h, w = img.shape[:2]
        ch, cw = crop_size, crop_size
        if h == ch and w == cw:
            return img, mask
        top = random.randint(0, h - ch)
        left = random.randint(0, w - cw)
        img = img[top:top+ch, left:left+cw]
        mask = mask[top:top+ch, left:left+cw]
        return img, mask

    def random_flip(self, img, mask):
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        if random.random() > 0.5:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
        return img, mask

    def random_rotate(self, img, mask):
        angle = random.choice([0, 90, 180, 270])
        if angle == 0:
            return img, mask
        img = np.rot90(img, k=angle//90)
        mask = np.rot90(mask, k=angle//90)
        return img, mask

    def add_gaussian_noise(self, img):
        if random.random() > 0.5:
            mean = 0
            std = random.uniform(0.01, 0.05)
            noise = np.random.normal(mean, std, img.shape)
            img = img + noise
            img = np.clip(img, 0, 1)
        return img

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.image_dir, img_name + '.tif')
        mask_path = os.path.join(self.mask_dir, img_name + '.png')

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 统一为3通道
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 统一尺寸
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # 标准化
        img = img / 255.0
        mask = mask / 255.0
        mask = (mask > 0.5).astype(np.float32)  # 二值化

        # 数据增强
        if self.mode == 'train':
            if random.random() > 0.5:
                img, mask = self.random_crop(img, mask, self.img_size)
            img, mask = self.random_flip(img, mask)
            img, mask = self.random_rotate(img, mask)
            img = self.add_gaussian_noise(img)

        # 转为tensor
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = torch.from_numpy(img.copy()).float()
        mask = torch.from_numpy(mask.copy()).unsqueeze(0).float()  # 1,H,W
        return img, mask


def split_dataset(image_dir, mask_dir, ratio=(0.7, 0.2, 0.1), seed=42):
    """
    按7:2:1划分训练、验证、测试集
    返回：train_list, val_list, test_list
    """
    all_images = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.tif')]
    all_images.sort()
    random.seed(seed)
    random.shuffle(all_images)
    n = len(all_images)
    n_train = int(n * ratio[0])
    n_val = int(n * ratio[1])
    train_list = all_images[:n_train]
    val_list = all_images[n_train:n_train+n_val]
    test_list = all_images[n_train+n_val:]
    return train_list, val_list, test_list

if __name__ == "__main__":
    # 训练集和验证集
    train_img_dir = 'data/Training/Tissue Images'
    train_mask_dir = 'data/Training/Masks'
    train_list, val_list, _ = split_dataset(train_img_dir, train_mask_dir)
    print(f"训练集: {len(train_list)}，验证集: {len(val_list)}")

    train_dataset = MedicalSegmentationDataset(train_img_dir, train_mask_dir, train_list, mode='train')
    val_dataset = MedicalSegmentationDataset(train_img_dir, train_mask_dir, val_list, mode='val')

    # 测试集
    test_img_dir = 'data/Test'
    test_mask_dir = 'data/Test/Masks'
    test_list = [os.path.splitext(f)[0] for f in os.listdir(test_img_dir) if f.endswith('.tif')]
    print(f"测试集: {len(test_list)}")
    test_dataset = MedicalSegmentationDataset(test_img_dir, test_mask_dir, test_list, mode='test')

    # 简单可视化
    import matplotlib.pyplot as plt
    for img, mask in train_dataset:
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.imshow(img.permute(1,2,0).numpy())
        plt.title('图像')
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(mask.squeeze(0).numpy(), cmap='gray')
        plt.title('Mask')
        plt.axis('off')
        plt.show()
        break
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from model.DeepLabV3Plus import DeepLabV3Plus
from utils.dataset import MedicalSegmentationDataset, split_dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import platform

# 可调节数据增强强度的Dataset包装器
class AugStrengthDataset(MedicalSegmentationDataset):
    def __init__(self, *args, aug_strength=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug_strength = aug_strength
    def random_crop(self, img, mask, crop_size):
        if np.random.rand() < self.aug_strength:
            return super().random_crop(img, mask, crop_size)
        return img, mask
    def random_flip(self, img, mask):
        if np.random.rand() < self.aug_strength:
            return super().random_flip(img, mask)
        return img, mask
    def random_rotate(self, img, mask):
        if np.random.rand() < self.aug_strength:
            return super().random_rotate(img, mask)
        return img, mask
    def add_gaussian_noise(self, img):
        if np.random.rand() < self.aug_strength:
            return super().add_gaussian_noise(img)
        return img

def train_net(
    net, device,
    train_img_dir, train_mask_dir,
    val_img_dir, val_mask_dir,
    epochs=50, batch_size=4, lr=1e-3, aug_strength=1.0
):
    os.makedirs('results', exist_ok=True)  # 确保保存目录存在
    # 划分训练/验证集
    train_list, val_list, _ = split_dataset(train_img_dir, train_mask_dir)
    # Dataset
    train_dataset = AugStrengthDataset(train_img_dir, train_mask_dir, train_list, mode='train', aug_strength=aug_strength)
    val_dataset = MedicalSegmentationDataset(val_img_dir, val_mask_dir, val_list, mode='val')
    # DataLoader
    is_windows = platform.system() == 'Windows'
    num_workers = 0 if is_windows else 4
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    # 优化器、损失函数、调度器
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()
    # 记录损失
    train_losses, val_losses = [], []
    # 训练
    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            outputs = outputs.squeeze(1)  # [B,H,W]
            loss = criterion(outputs, masks.squeeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f'Epoch {epoch+1}/{epochs} 训练损失: {avg_train_loss:.4f}', end='; ')
        # 验证
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = net(images)
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, masks.squeeze(1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f'验证损失: {avg_val_loss:.4f}')
        # 学习率调度
        scheduler.step(avg_val_loss)
        print(f'  当前学习率: {optimizer.param_groups[0]["lr"]:.6f}')
        # 保存模型
        torch.save(net.state_dict(), 'results/final_model.pth')
    # 绘制损失曲线
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, 'b-', label='训练损失')
    plt.plot(range(1, len(val_losses)+1), val_losses, 'r-', label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练与验证损失曲线')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/loss_curve.png')
    print('损失曲线已保存至 results/loss_curve.png')
    return net

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    # DeepLabV3Plus模型
    net = DeepLabV3Plus(in_channels=3, num_classes=1)
    net.to(device)
    # 数据路径
    train_img_dir = 'data/Training/Tissue Images'
    train_mask_dir = 'data/Training/Masks'
    val_img_dir = train_img_dir
    val_mask_dir = train_mask_dir
    # 训练
    trained_net = train_net(
        net, device,
        train_img_dir, train_mask_dir,
        val_img_dir, val_mask_dir,
        epochs=50, batch_size=4, lr=1e-3, aug_strength=1.0
    )
    print('训练完成！')
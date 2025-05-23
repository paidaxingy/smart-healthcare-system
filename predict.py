import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from model.DeepLabV3Plus import DeepLabV3Plus
from utils.dataset import MedicalSegmentationDataset
from torch.utils.data import DataLoader

# Dice系数
def dice_coef(pred, target, threshold=0.5, eps=1e-6):
    pred = (pred > threshold).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum(dim=(1,2))
    union = pred.sum(dim=(1,2)) + target.sum(dim=(1,2))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()

# IoU
def iou_score(pred, target, threshold=0.5, eps=1e-6):
    pred = (pred > threshold).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum(dim=(1,2))
    union = (pred + target).clamp(0,1).sum(dim=(1,2))
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    # 加载模型
    net = DeepLabV3Plus(in_channels=3, num_classes=1)
    net.to(device)
    net.load_state_dict(torch.load('results/final_model.pth', map_location=device))
    net.eval()
    # 加载测试集
    test_img_dir = 'data/Test'
    test_mask_dir = 'data/Test/Masks'
    test_list = [os.path.splitext(f)[0] for f in os.listdir(test_img_dir) if f.endswith('.tif')]
    test_dataset = MedicalSegmentationDataset(test_img_dir, test_mask_dir, test_list, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"测试集大小: {len(test_dataset)}")
    dices, ious = [], []
    os.makedirs('results/vis', exist_ok=True)
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = net(images)
            preds = torch.sigmoid(outputs).cpu().squeeze(1)  # [B,H,W]
            masks = masks.cpu().squeeze(1)
            # 计算指标
            dice = dice_coef(preds, masks)
            iou = iou_score(preds, masks)
            dices.append(dice)
            ious.append(iou)
            # 可视化前5张
            if idx < 5:
                plt.figure(figsize=(12,4))
                plt.subplot(1,3,1)
                plt.imshow(images.cpu().squeeze().permute(1,2,0).numpy())
                plt.title('原图')
                plt.axis('off')
                plt.subplot(1,3,2)
                plt.imshow(masks.squeeze().numpy(), cmap='gray')
                plt.title('真实Mask')
                plt.axis('off')
                plt.subplot(1,3,3)
                plt.imshow((preds.squeeze().numpy()>0.5).astype(np.float32), cmap='gray')
                plt.title('预测Mask')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f'results/vis/test_{idx}.png')
                plt.close()
    print(f"平均Dice系数: {np.mean(dices):.4f}")
    print(f"平均IoU: {np.mean(ious):.4f}")
    with open('results/test_metrics.txt', 'w', encoding='utf-8') as f:
        f.write(f"平均Dice系数: {np.mean(dices):.4f}\n")
        f.write(f"平均IoU: {np.mean(ious):.4f}\n")
    print('部分分割结果已保存至 results/vis/')
    print('评估指标已保存至 results/test_metrics.txt')
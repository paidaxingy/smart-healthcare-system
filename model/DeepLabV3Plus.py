import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- ResNet50主干的Bottleneck残差块 -----------------
class Bottleneck(nn.Module):
    """ResNet50的瓶颈残差块"""
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# ----------------- ASPP模块 -----------------
class ASPP(nn.Module):
    """空洞空间金字塔池化模块（Atrous Spatial Pyramid Pooling）"""
    def __init__(self, in_channels, out_channels, rates=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.blocks = nn.ModuleList()
        for rate in rates:
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        # 全局平均池化分支
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        aspp_outs = [block(x) for block in self.blocks]
        # 全局池化分支
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=x.shape[2:], mode='bilinear', align_corners=True)
        aspp_outs.append(gp)
        x = torch.cat(aspp_outs, dim=1)
        x = self.out_conv(x)
        return x

# ----------------- DeepLabv3+主结构（ResNet50主干） -----------------
class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        """
        DeepLabv3+分割模型，主干为ResNet50
        :param in_channels: 输入通道数，默认为3（RGB）
        :param num_classes: 输出类别数，默认为1（二值分割）
        """
        super(DeepLabV3Plus, self).__init__()
        # --- 编码器：ResNet50主干 ---
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)   # 输出1/4
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)  # 输出1/8
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)  # 输出1/16
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)  # 输出1/32

        # --- ASPP模块 ---
        self.aspp = ASPP(2048, 256)  # 2048=512*4

        # --- 解码器 ---
        self.reduce_low = nn.Sequential(
            nn.Conv2d(512, 48, 1, bias=False),  # 512=128*4
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256+48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """构建残差层"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # 编码器
        x0 = F.relu(self.bn1(self.conv1(x)))   # [B,64,H/2,W/2]
        x0 = self.maxpool(x0)                  # [B,64,H/4,W/4]
        x1 = self.layer1(x0)                   # [B,256,H/4,W/4]
        x2 = self.layer2(x1)                   # [B,512,H/8,W/8]  (低级特征)
        x3 = self.layer3(x2)                   # [B,1024,H/16,W/16]
        x4 = self.layer4(x3)                   # [B,2048,H/32,W/32]

        # ASPP
        aspp_out = self.aspp(x4)               # [B,256,H/32,W/32]
        aspp_out = F.interpolate(aspp_out, size=x2.shape[2:], mode='bilinear', align_corners=True)

        # 解码器
        low_feat = self.reduce_low(x2)         # [B,48,H/8,W/8]
        concat = torch.cat([aspp_out, low_feat], dim=1)  # [B,304,H/8,W/8]
        out = self.decoder(concat)             # [B,num_classes,H/8,W/8]
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out

    def get_parameters(self):
        """获取模型所有参数的名称和大小"""
        params = []
        total_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                param_size = list(param.size())
                param_count = torch.prod(torch.tensor(param_size)).item()
                total_params += param_count
                params.append((name, param_size, param_count))
        return params, total_params

if __name__ == '__main__':
    # 创建DeepLabv3+模型实例（3通道输入，单通道输出）
    model = DeepLabV3Plus(in_channels=3, num_classes=1)
    model.eval()
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        y = model(x)
    print(f"\n输入尺寸: {x.size()}")
    print(f"输出尺寸: {y.size()}")
    # 获取并打印模型参数
    params, total = model.get_parameters()
    print(f"\n模型参数详情:")
    for name, size, count in params:
        print(f"{name}: {size}, 参数数量: {count:,}")
    print(f"\n总参数数量: {total:,}") 
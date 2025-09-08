import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """A tiny residual block: 2x (3x3 conv + BN), with optional downsample."""
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = F.relu(out + identity, inplace=True)
        return out


class ResNet10(nn.Module):
    """
    Lightweight ResNet-10 (for 84x84/80x80 few-shot). Returns a feature vector (no classifier here).
    Compatible with the wrappers: forward(x, is_support=False) -> embeddings
    """
    def __init__(self, exp_dict):
        super().__init__()
        # You can control width via exp_dict if you want (optional)
        base = int(exp_dict.get("resnet10_width", 64))
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        # 4 stages, one BasicBlock per stage => 8 conv layers + 2 in stem/final = ~10
        self.layer1 = BasicBlock(base,      base,      stride=1)  # 80x80
        self.layer2 = BasicBlock(base,      base * 2,  stride=2)  # 40x40
        self.layer3 = BasicBlock(base * 2,  base * 4,  stride=2)  # 20x20
        self.layer4 = BasicBlock(base * 4,  base * 8,  stride=2)  # 10x10

        self.out_dim = base * 8
        p = float(exp_dict.get("dropout", 0.0))
        self._use_drop = p > 0.0
        self.drop = nn.Dropout(p=p) if self._use_drop else nn.Identity()

        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, is_support: bool = False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.drop(x) if self._use_drop else x
        return x

import torch
from torch import nn
import torchvision

class CANet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        bits = self.config.code_length
        classlen = self.config.classlen

        if self.config.model_name == 'resnet50':
            self.backbone = torchvision.models.resnet50(pretrained=True)
            self.backbone.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            # Adaptive hyper
            self.alpha1 = nn.Parameter(torch.ones(1) * 1, requires_grad=True)
            self.alpha2 = nn.Parameter(torch.ones(1) * 1, requires_grad=True)

            self.backbone.b1 = nn.Linear(1024, bits)
            self.backbone.b2 = nn.Linear(1024, bits)
            self.backbone.b3 = nn.Linear(1024, bits)
            self.backbone.b_cat = nn.Linear(3072, bits)

            self.backbone.fc = nn.Linear(bits, classlen)

            self.relu = nn.ReLU(inplace=True)
            self.num_ftrs = 2048
            self.feature_size = 512
            self.backbone.fc_x = nn.Linear(self.num_ftrs, classlen)

            # stage 1
            self.backbone.conv_block1 = nn.Sequential(
                BasicConv(self.num_ftrs // 4, self.feature_size, kernel_size=1, stride=1, padding=0, relu=True),
                BasicConv(self.feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True),
                nn.AdaptiveMaxPool2d(1),
            )

            # stage 2
            self.backbone.conv_block2 = nn.Sequential(
                BasicConv(self.num_ftrs // 2, self.feature_size, kernel_size=1, stride=1, padding=0, relu=True),
                BasicConv(self.feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True),
                nn.AdaptiveMaxPool2d(1),
            )

            # stage 3
            self.backbone.conv_block3 = nn.Sequential(
                BasicConv(self.num_ftrs, self.feature_size, kernel_size=1, stride=1, padding=0, relu=True),
                BasicConv(self.feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True),
                nn.AdaptiveMaxPool2d(1),
            )

            # concat features from different stages
            self.backbone.hashing_concat = nn.Sequential(
                nn.BatchNorm1d(self.num_ftrs // 2 * 3, affine=True),
                nn.Linear(self.num_ftrs // 2 * 3, self.feature_size),
                nn.BatchNorm1d(self.feature_size, affine=True),
                nn.ELU(inplace=True),
                nn.Linear(self.feature_size, bits),
            )

    def forward(self, x):
        return self.forward_vanilla(x)

    def forward_vanilla(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x1 = self.backbone.maxpool(x)
        x2 = self.backbone.layer1(x1)
        f1 = self.backbone.layer2(x2)
        f2 = self.backbone.layer3(f1)
        f3 = self.backbone.layer4(f2)

        feats = f3

        f11 = self.backbone.conv_block1(f1).view(-1, self.num_ftrs // 2)
        f11_b = self.backbone.b1(f11)

        f22 = self.backbone.conv_block2(f2).view(-1, self.num_ftrs // 2)
        f22_b = self.backbone.b2(f22)

        f33 = self.backbone.conv_block3(f3).view(-1, self.num_ftrs // 2)
        f33_b = self.backbone.b3(f33)
        y33 = self.backbone.fc(f33_b)

        f44 = torch.cat((f11, f22, f33), -1)
        f44_b = self.backbone.hashing_concat(f44)

        # x = self.backbone.avgpool(feats)
        # x = torch.flatten(x, 1)
        # y_x = self.backbone.fc_x(x)

        return self.alpha1, self.alpha2, f44_b, y33, feats



# Model Setting
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
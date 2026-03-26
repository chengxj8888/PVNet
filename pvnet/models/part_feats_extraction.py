import torch
import torch.nn as nn
import numpy as np

__all__ = ['Part_feat_Extraction']


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride, padding=padding, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.net(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, padding=1,  dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride, padding=padding, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(outc, outc, kernel_size=ks, dilation=dilation, stride=1, padding=padding, bias=False),
        )

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                nn.Conv1d(inc, outc, kernel_size=1, dilation=1, stride=stride, bias=False),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        net_x = self.net(x)             # (18000, 32)
        down_x = self.downsample(x)     # (18000, 32)
        out = self.relu(net_x + down_x)
        return out


class Part_feat_Extraction(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        cr = kwargs.get('cr', 1.0)
        in_channels = kwargs.get('in_channels', 3)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.embed_dim = cs[-1]
        self.run_up = kwargs.get('run_up', True)
       
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, cs[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(cs[0], cs[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=3, stride=1, padding=1, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, padding=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, padding=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, padding=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, padding=1, dilation=1),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, padding=1, dilation=1),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, padding=1, dilation=1),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, padding=1, dilation=1),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, padding=1, dilation=1),
        )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.permute(0,2,1)      
        
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)            

        return x4.permute(0,2,1)        

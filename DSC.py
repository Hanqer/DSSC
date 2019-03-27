import torch
import os
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.nn import init
from torch.nn import Parameter
from torchvision.utils import save_image

class DSC(nn.Module):
    def __init__(self, in_channels, feat_size=None):
        super(DSC, self).__init__()
        self.feat_conv = nn.Sequential(nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, 1, 1),
            nn.Sigmoid())
        self.context_conv1 = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.irnn1 = MMLayer(feat_size)
        self.context_conv2 = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.irnn2 = MMLayer(feat_size)
        self.context_conv3 = nn.Conv2d(in_channels, in_channels, 1)
    def forward(self, x):
        xx = self.feat_conv(x)
        con = self.context_conv1(x)
        con = self.irnn1(con)
        con = con * xx
        con = self.context_conv2(con)
        con = self.irnn2(con)
        con = con * xx
        con = self.context_conv3(con)
        return con + x
        

class MMLayer(nn.Module):
    def __init__(self, size):
        super(MMLayer, self).__init__()
        self.size = size
        # print(size)
        self.w_h = nn.Parameter(torch.Tensor(size, size))
        self.w_v = nn.Parameter(torch.Tensor(size, size))
        self.gamma = nn.Parameter(torch.Tensor(1))

        self.w_h.data[...] = torch.eye(size)
        self.w_v.data[...] = torch.eye(size)
        self.gamma.data[...] = 1

    def forward(self, x):
        N, C, H, W = x.shape
        assert self.size == W == H
        x1 = torch.mm(x.view(N*C*H, W), self.w_h).view(N, C, H, W)
        x2 = torch.mm(x.permute(0, 1, 3, 2).contiguous().view(N*C*W, H), self.w_v).view(N, C, W, H).permute(0, 1, 3, 2)

        return F.relu(torch.cat((x1, x2), dim=1))



    
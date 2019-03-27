import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt as bwdist
def Bwdist(image, t=8):
    if not isinstance(image, np.ndarray):
        img = np.array(image)
        # print(img.max())
    else:
        img = image
    dist1 = bwdist(np.logical_not(img))
    dist2 = bwdist(img)
    dist1 = 255 - ((dist1<t)*255).astype(np.uint8)
    dist2 = 255 - ((dist2<t)*255).astype(np.uint8)
    b = Image.fromarray(dist1).convert('L')
    f = Image.fromarray(dist2).convert('L')
    return b, f

class Bgfuse(nn.Module):
    def __init__(self):
        super(Bgfuse, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x, bg):
        bg = F.interpolate(bg, x.size()[2:], mode='nearest')
        assert bg.requires_grad == False
        assert bg.size(1) == 1
        ratio = bg.numel() / bg.sum() 
        bg_feat = x * bg
        bg_feat = self.pool(bg_feat).view(x.size(0), -1, 1, 1)
        bg_feat *= ratio
        bg_feat = bg_feat.expand_as(x)
        #bg_feat = F.interpolate(bg_feat, x.size()[2:], mode='bilinear', align_corners=True)
        return torch.cat((x, bg_feat), dim = 1)

class BgAttention(nn.Module):
    def __init__(self):
        super(BgAttention, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x, bg):
        bg = F.interpolate(bg, x.size()[2:], mode='nearest')
        assert bg.requires_grad == False
        assert bg.size(1) == 1
        ratio = bg.numel() / bg.sum() 
        bg_feat = x * bg
        bg_feat = self.pool(bg_feat).view(x.size(0), -1, 1, 1)
        bg_feat *= ratio
        bg_feat = bg_feat.expand_as(x)

        affinity = x * bg_feat
        affinity = self.relu(affinity)

        aff_feat = x * affinity
        return aff_feat + x


class BF_Attention(nn.Module):
    def __init__(self, in_channels):
        super(BF_Attention, self).__init__()
        self.conv_f = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_value = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_fb = nn.Conv2d(in_channels, in_channels, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, bg, fg, normlize=False):
        N, C, H, W = x.size()
        bg = F.interpolate(bg, x.size()[2:], mode='nearest')
        fg = F.interpolate(fg, x.size()[2:], mode='nearest')
        bg_ratio = bg.numel() / bg.sum()
        fg_ratio = fg.numel() / fg.sum()
        bf_feat = self.conv_fb(x)

        bg_feat = (bf_feat * bg) * bg_ratio
        fg_feat = (bf_feat * fg) * fg_ratio

        bg_feat = self.pool(bg_feat).view(N, -1, 1)
        fg_feat = self.pool(fg_feat).view(N, -1, 1)
        bf_feat = torch.cat((bg_feat, fg_feat), dim = 2) # N, C, 2
        x_value = self.conv_value(x).view(N, -1, H*W).permute(0, 2, 1) # N, HW, C
        if normlize:
            x_value = torch.nn.functional.normalize(x_value, p=2, dim=2)
            bf_feat = torch.nn.functional.normalize(bf_feat, p=2, dim=1)
        attention = torch.bmm(x_value, bf_feat) # N, HW, 2
        attention = torch.softmax(attention, dim = 2)
        out = torch.bmm(bf_feat, attention.permute(0, 2, 1)).view(N, C, H, W) # N, C, H, W
        return self.conv_f(x) + self.gamma * out


if __name__ == "__main__":
    img = Image.open('/home/hanqi/dataset/DUTS/DUTS-TE/DUTS-TE-Mask/ILSVRC2012_test_00000003.png').convert('L')
    img.show()
    b, f = Bwdist(img)
    b.show()
    f.show()
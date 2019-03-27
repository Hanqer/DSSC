import torch
import time
import numpy as np
from torch import nn
from torch.nn import init
from DSC import DSC
from bg import Bgfuse, BgAttention, BF_Attention
from PIL import Image
from scipy.ndimage import distance_transform_edt as bwdist
from torchvision.utils import save_image
import os
import cv2
# vgg choice
base = {'dss': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}
# extend vgg choice --- follow the paper, you can change it
extra = {'dss': [(64, 128, 3, [8, 16, 32, 64]), (128, 128, 3, [4, 8, 16, 32]), (256, 256, 5, [8, 16]),
                 (512, 256, 5, [4, 8]), (512, 512, 5, []), (512, 512, 7, [])]}
connect = {'dss': [[2, 3, 4, 5], [2, 3, 4, 5], [4, 5], [4, 5], [], []]}
feat_size = [256, 128, 64, 32, 16, 8]

def Bwdist(image, t=8):
    shape = image.shape[2:]
    batch_size = image.shape[0]
    image = image.reshape((-1, shape[0], shape[1], 1))
    # image = image.squeeze()
    image2 = image.copy()
    dist1 = image.copy()
    dist2 = image.copy()
    for i in range(batch_size):
        image[i,:,:,:] = bwdist(np.logical_not(image[i,:,:,:]))
        image2[i,:,:,:] = bwdist((image2[i,:,:,:]))
        dist1[i,:,:,:] = 1.0 - ((image[i,:,:,:]<t)*1.0)
        dist2[i,:,:,:] = 1.0 - ((image2[i,:,:,:]<t)*1.0)
    dist1 = dist1.reshape((-1, 1, shape[0], shape[1]))
    dist2 = dist2.reshape((-1, 1, shape[0], shape[1]))
    
    return dist1, dist2

# vgg16
def vgg(cfg, i=3, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers


# feature map before sigmoid: build the connection and deconvolution
class ConcatLayer(nn.Module):
    def __init__(self, list_k, k, scale=True):
        super(ConcatLayer, self).__init__()
        l, up, self.scale = len(list_k), [], scale
        for i in range(l):
            up.append(nn.ConvTranspose2d(1, 1, list_k[i], list_k[i] // 2, list_k[i] // 4))
        self.upconv = nn.ModuleList(up)
        self.conv = nn.Conv2d(l + 1, 1, 1, 1)
        self.deconv = nn.ConvTranspose2d(1, 1, k * 2, k, k // 2) if scale else None

    def forward(self, x, list_x):
        elem_x = [x]
        for i, elem in enumerate(list_x):
            elem_x.append(self.upconv[i](elem))
        if self.scale:
            out = self.deconv(self.conv(torch.cat(elem_x, dim=1)))
        else:
            out = self.conv(torch.cat(elem_x, dim=1))
        return out


# extend vgg: side outputs
class FeatLayer(nn.Module):
    def __init__(self, in_channel, channel, k):
        super(FeatLayer, self).__init__()
        self.main = nn.Sequential(nn.Conv2d(in_channel, channel, k, 1, k // 2), nn.ReLU(inplace=True),
                                  nn.Conv2d(channel, channel, k, 1, k // 2), nn.ReLU(inplace=True)
                                    )
        self.last_conv = nn.Conv2d(channel, 1, 1, 1)

    def forward(self, x):
        xx = self.main(x)
        x = self.last_conv(xx)
        return x, xx

class BgLayer(nn.Module):
    def __init__(self, in_channel, channel, k, feat_size):
        super(BgLayer, self).__init__()
        self.bg = BF_Attention(channel)
        self.last_conv = nn.Conv2d(channel, 1, 1, 1)
    def forward(self, x, bg, fg):
        x = self.bg(x, bg, fg, normlize=False)
        x = self.last_conv(x)
        return x

# fusion features
class FusionLayer(nn.Module):
    def __init__(self, nums=6):
        super(FusionLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(nums))
        self.nums = nums
        self._reset_parameters()

    def _reset_parameters(self):
        init.constant_(self.weights, 1 / self.nums)

    def forward(self, x):
        for i in range(self.nums):
            out = self.weights[i] * x[i] if i == 0 else out + self.weights[i] * x[i]
        return out


# extra part
def extra_layer(vgg, cfg):
    feat_layers, concat_layers, concat_layers2, bg_layers, scale = [], [], [], [], 1
    feat_size = 256
    for k, v in enumerate(cfg):
        # side output (paper: figure 3)
        feat_layers += [FeatLayer(v[0], v[1], v[2])]
        bg_layers += [BgLayer(v[0], v[1], v[2], feat_size)]
        # feature map before sigmoid
        concat_layers += [ConcatLayer(v[3], scale, k != 0)]
        concat_layers2 += [ConcatLayer(v[3], scale, k != 0)]
        scale *= 2
        feat_size = feat_size // 2
    return vgg, feat_layers, concat_layers, bg_layers, concat_layers2


# DSS network
# Note: if you use other backbone network, please change extract
class DSS(nn.Module):
    def __init__(self, base, feat_layers, concat_layers, bg_layers, concat_layers2, connect, extract=[3, 8, 15, 22, 29], v2=True):
        super(DSS, self).__init__()
        self.extract = extract
        self.connect = connect
        self.base = nn.ModuleList(base)
        self.feat = nn.ModuleList(feat_layers)
        self.bg_feat = nn.ModuleList(bg_layers)
        self.comb = nn.ModuleList(concat_layers)
        self.comb2 = nn.ModuleList(concat_layers2)
        self.pool = nn.AvgPool2d(3, 1, 1)
        self.v2 = v2
        self.select = [1, 2, 3, 6]
        if v2: self.fuse = FusionLayer()

    def forward(self, x, bg=None, fg=None, mode='train'):
        prob, back, y, feats, num = list(), list(), list(), list(), 0
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.extract:
                feat0, feat1 = self.feat[num](x)
                y.append(feat0)
                feats.append(feat1)
                num += 1
        # side output
        feat0, feat1 = self.feat[num](self.pool(x))
        y.append(feat0)
        feats.append(feat1)
        assert feats[0].size(1) == 128
        #concat
        for i, k in enumerate(range(len(y))):
            back.append(self.comb[i](y[i], [y[j] for j in self.connect[i]]))
        #fuse
        back.append(self.fuse(back))
        #generate bg
        for i in back: prob.append(torch.sigmoid(i))
        if mode=='test':
            result = torch.mean(torch.cat([prob[i] for i in self.select], dim=1), dim=1, keepdim=True)
            result = (result > 0.5).float()
            result = result.detach().cpu().numpy()
            fork_bg, fork_fg = Bwdist(result)
            fork_bg = torch.from_numpy(fork_bg).cuda()
            fork_fg = torch.from_numpy(fork_fg).cuda()
            assert fork_bg.size(1) == 1
            # save_image(fork_bg, '/home/hanqi/fork_bg.png', normalize=True)
            # save_image(fork_fg, '/home/hanqi/fork_fg.png', normalize=True)
            # fork_bg.error()
            # assert result
        
        yy, back2 = list(), list()
        #side output
        for i, f in enumerate(feats):
            if mode == 'test':
                yy.append(self.bg_feat[i](f, fork_bg, fork_fg))
            else:
                #label = torch.zeros((8, 1, 256, 256)).cuda()
                yy.append(self.bg_feat[i](f, bg, fg))
        #concat
        for i in range(len(yy)):
            back2.append(self.comb2[i](yy[i], [yy[j] for j in self.connect[i]]))
        # fusion map
        if self.v2:
            # version2: learning fusion
            back2.append(self.fuse(back2))
        else:
            # version1: mean fusion
            back2.append(torch.cat(back, dim=1).mean(dim=1, keepdim=True))
        # add sigmoid
        res = list()
        for i in back2: res.append(torch.sigmoid(i))
        res.extend(prob)
        assert len(res) == 14
        return res


# build the whole network
def build_model():
    return DSS(*extra_layer(vgg(base['dss'], 3), extra['dss']), connect['dss'])


# weight init
def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


if __name__ == '__main__':
    net = build_model()
    img = torch.randn(1, 3, 64, 64)
    net = net.to(torch.device('cuda:0'))
    img = img.to(torch.device('cuda:0'))
    out = net(img)
    k = [out[x] for x in [1, 2, 3, 6]]
    print(len(k))
    # for param in net.parameters():
    #     print(param)

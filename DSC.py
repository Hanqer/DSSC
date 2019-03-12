import torch
import time
import torch.nn as nn
from torch.optim import SGD
from torch.nn import init
from torch.nn import Parameter
class DSC(nn.Module):
    def __init__(self, in_channels, feat_width=None, feat_height=None):
        super(DSC, self).__init__()
        self.feat_conv = nn.Sequential(nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.ReLU(),
            # nn.Conv2d(128, 128, 3, padding=1),
            # nn.ReLU(),
            nn.Conv2d(128, 4, 1),
            nn.Sigmoid())
        self.context_conv1 = nn.Conv2d(in_channels, 128, 1)
        self.irnn1 = IRNN(128)
        self.eltwise = Eltwise(128*4, 4)
        # self.context_conv2 = nn.Conv2d(512, 128, 1)
        # self.irnn2 = IRNN(128)
        self.context_conv3 = nn.Conv2d(512, 128, 1)
    def forward(self, x):
        xx = self.feat_conv(x)
        con = self.context_conv1(x)
        con = self.irnn1(con)
        con = self.eltwise(con, xx)
        # con = self.context_conv2(con)
        # con = self.irnn2(con)
        # con = self.eltwise(con, xx)
        con = self.context_conv3(con)
        return torch.cat((con, x), dim=1)

class Eltwise(nn.Module):
    def __init__(self, in_channels, mask_channels):
        super(Eltwise, self).__init__()
        self.in_channels = in_channels
        self.mask_channels = mask_channels

    def forward(self, x, mask):
        p_channels = self.in_channels // self.mask_channels
        n, c, w, h = x.shape
        for i in range(self.mask_channels):
            x[:, i*p_channels:(i+1)*p_channels, :, :] = x[:, i*p_channels:(i+1)*p_channels, :, :].clone() * mask[:, i, :, :].view(n, 1, w, h)
        return x


class Irnn(nn.Module):
    def __init__(self, in_channels):
        super(Irnn, self).__init__()
        self.in_channels = in_channels
        self.weight = Parameter(torch.eye(in_channels))

    def forward(self, x):
        N, C, W, H = x.shape
        output0 = x.clone()
        for i in range(1, W):
            data1 = output0[:, :, :, i-1]
            # N, C, H --> N, H, C
            data1 = torch.transpose(data1, 1, 2).contiguous()
            data1 = data1.view(-1, self.in_channels)
            tmp = torch.mm(data1, self.weight)
            tmp = torch.reshape(tmp, (N, H, C))
            # N, H, C --> N, C, H
            tmp = torch.transpose(tmp, 1, 2).contiguous()
            output0[:, :, :, i] += tmp

        output1 = x.clone()
        for i in range(W-2, -1, -1):
            data1 = output1[:, :, :, i+1]
            # N, C, H --> N, H, C
            data1 = torch.transpose(data1, 1, 2).contiguous()
            data1 = data1.view(-1, self.in_channels)
            tmp = torch.mm(data1, self.weight)
            tmp = torch.reshape(tmp, (N, H, C))
            # N, H, C --> N, C, H
            tmp = torch.transpose(tmp, 1, 2).contiguous()
            output1[:, :, :, i] += tmp

        output2 = x.clone()
        for i in range(1, H):
            # data of previous row
            data1 = output2[:, :, i-1, :]
            # N, C, W --> N, W, C
            data1 = torch.transpose(data1, 1, 2).contiguous()
            data1 = data1.view(-1, self.in_channels)
            tmp = torch.mm(data1, self.weight)
            tmp = torch.reshape(tmp, (N, H, C))
            # N, W, C --> N, C, W
            tmp = torch.transpose(tmp, 1, 2).contiguous()
            output2[:, :, i, :] += tmp
        
        output3 = x.clone()
        for i in range(H-2, -1, -1):
            # data of previous row
            data1 = output3[:, :, i+1, :]
            # N, C, W --> N, W, C
            data1 = torch.transpose(data1, 1, 2).contiguous()
            data1 = data1.view(-1, self.in_channels)
            tmp = torch.mm(data1, self.weight)
            tmp = torch.reshape(tmp, (N, H, C))
            # N, W, C --> N, C, W
            tmp = torch.transpose(tmp, 1, 2).contiguous()
            output3[:, :, i, :] += tmp
        return torch.cat((output0, output1, output2, output3), dim=1)

class IRNN(nn.Module):
    def __init__(self, input_size):
        
        super(IRNN, self).__init__()

        self.horizontal = nn.RNN(input_size=input_size, hidden_size=input_size, bidirectional=True)
        self.vertical = nn.RNN(input_size=input_size, hidden_size=input_size, bidirectional=True)

    def forward(self, data):
        
        N, C, H, W = data.shape
        data_h = data.permute(3, 0, 2, 1).contiguous().view(W, N*H, C)
        data_v = data.permute(2, 0, 3, 1).contiguous().view(H, N*W, C)
        
        output_h = self.horizontal(data_h)[0]
        output_v = self.vertical(data_v)[0]

        output_h = output_h.view(W, N, H, C*2).permute(1, 3, 2, 0).contiguous()
        output_v = output_h.view(H, N, W, C*2).permute(1, 3, 0, 2).contiguous()

        return torch.cat((output_h, output_v), dim=1)

if __name__ == '__main__':
    #test irnn model
    input_size = 128
    model = IRNN(input_size)

    x = torch.ones((1, input_size, 5, 5))
    output = model(x)
    # print(output)

    #test Eltwise
    # mask = torch.ones((1, 4, 5, 5))
    # mask[:,2,:,:] *= 2
    # model2 = Eltwise(3*4, 4)
    # output = model2(output, mask)
    # # print(output)

    # #test DSC
    # model3 = DSC(256)
    # x = torch.rand((2, 256, 16, 16))
    # y = model3(x)
    # print(y.shape)

    
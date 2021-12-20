import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os
import pdb

base_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
upsample_cfg = ['U', [256, 128], 'U', [128, 64], 'U', [16, 16], 'U', [16, 16]]
erf_cfg = [3, 'C', 16, 'C', 64, 64, 64, 64, 'C', 128, 128, 128, 'C', 256]

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=(1, 1), groups=1, relu=True, bn=True, bias=False, ring_pad=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.ring_pad = ring_pad
        self.kernel_size = kernel_size
        self.pad_size = 0
        if isinstance(self.kernel_size, int):
            self.pad_size = self.kernel_size // 2 + dilation[1] - 1
        elif isinstance(self.kernel_size, tuple):
            self.pad_size = self.kernel_size[1] // 2 + dilation[1] - 1

    def forward(self, x):
        begin, end = 0, x.shape[3] + 2 * self.pad_size
        if self.ring_pad and self.pad_size:
            left_pad = x[:, :, :, -self.pad_size: ]
            right_pad = x[:, :, :, : self.pad_size]
            x = torch.cat([left_pad, x, right_pad], dim=-1)
            begin += self.pad_size
            end -= self.pad_size

        x = self.conv(x)

        if self.ring_pad and self.pad_size:
            x = x[:, :, :, begin: end]
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)

        return x

class Scene(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(Scene, self).__init__()
        self.in_channels = in_planes
        self.out_channels = out_planes

        #self.conv1 = BasicConv(in_planes, out_planes // 2, kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Linear(in_planes, out_planes // 2)
        self.conv2 = BasicConv(in_planes, out_planes // 2, kernel_size=1, stride=1, padding=0)
        self.conv3_1 = BasicConv(in_planes, out_planes // 4, kernel_size=(3, 3), stride=1, padding=(1, 1), dilation=(1, 1))
        self.conv3_2 = BasicConv(in_planes, out_planes // 4, kernel_size=(3, 3), stride=1, padding=(1, 2), dilation=(1, 2))
        self.conv3_3 = BasicConv(in_planes, out_planes // 4, kernel_size=(3, 3), stride=1, padding=(1, 4), dilation=(1, 4))
        self.conv3_4 = BasicConv(in_planes, out_planes // 4, kernel_size=(3, 3), stride=1, padding=(2, 1), dilation=(2, 1))
        self.conv = BasicConv(out_planes * 2, out_planes * 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        #x1 = torch.mean(torch.mean(x, dim=3, keepdim=True), dim=2, keepdim=True)
        #x1 = self.conv1(x1)
        x1 = torch.mean(torch.mean(x, dim=3, keepdim=False), dim=2, keepdim=False)
        x1 = self.fc1(x1)
        x1 = x1.unsqueeze(-1).unsqueeze(-1)
        x1 = x1.repeat(1, 1, x.shape[2], x.shape[3])

        x2 = self.conv2(x)
        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(x)
        x3_3 = self.conv3_3(x)
        x3_4 = self.conv3_4(x)

        y = torch.cat([x1, x2, x3_1, x3_2, x3_3, x3_4], dim=1)
        y = self.conv(y)

        return y

class Upsample(nn.Module):

    def __init__(self, in_planes, out_planes, stride=2):
        super(Upsample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv2d = BasicConv(in_planes=in_planes, out_planes=out_planes, kernel_size=3, stride=1, padding=1, bn=False, relu=False)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv2d(x)
        return x

class SubPixelUpsample(nn.Module):

    def __init__(self, in_planes, out_planes, stride=2):
        super(SubPixelUpsample, self).__init__()
        self.conv1 = BasicConv(in_planes=in_planes, out_planes=in_planes * stride * stride, kernel_size=1, stride=1, padding=0, bn=False, relu=False)
        self.shuffle = nn.PixelShuffle(stride)
        self.conv2 = BasicConv(in_planes=in_planes, out_planes=out_planes, kernel_size=3, stride=1, padding=1, bn=False, relu=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.shuffle(x)
        x = self.conv2(x)
        return x

class ResidualBlock(nn.Module):

    def __init__(self, pre_planes, in_planes, stride=1, bn=True, relu=True):
        super(ResidualBlock, self).__init__()
        self.pre_conv = BasicConv(pre_planes, in_planes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, bn=False, relu=False)
        self.conv1 = BasicConv(in_planes, in_planes, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), groups=1, bias=False, bn=False, relu=False)
        self.conv2 = BasicConv(in_planes, in_planes, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), dilation=(1, 1), groups=1, bias=False, bn=True, relu=False)
        self.conv3 = BasicConv(in_planes, in_planes, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), groups=1, bias=False, bn=False, relu=False)
        self.conv4 = BasicConv(in_planes, in_planes, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), dilation=(1, 1), groups=1, bias=False, bn=True, relu=False)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.pre_conv(x)
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.relu(x + y)
        return y

class NonBottleBlock(nn.Module):

    def __init__(self, in_planes, stride=1, bn=True, relu=True):
        super(NonBottleBlock, self).__init__()
        self.conv1 = BasicConv(in_planes, in_planes, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), groups=1, bias=False, bn=False, relu=True)
        self.conv2 = BasicConv(in_planes, in_planes, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), dilation=(1, 1), groups=1, bias=False, bn=False, relu=True)
        self.conv3 = BasicConv(in_planes, in_planes, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), groups=1, bias=False, bn=False, relu=True)
        self.conv4 = BasicConv(in_planes, in_planes, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), dilation=(1, 1), groups=1, bias=False, bn=False, relu=False)
        self.bn = nn.BatchNorm2d(in_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        return self.relu(self.bn(x + self.conv4(self.conv3(self.conv2(self.conv1(x))))))

class get_disp(nn.Module):

    def __init__(self, num_in_layers):
        super(get_disp, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        return 0.1 * self.relu((2 * self.sigmoid(x) - 1))

def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=(1, 1))
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return layers

def erfnet(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    i = 1

    while i < len(cfg):
        if cfg[i] == 'C':
            layers += [BasicConv(cfg[i - 1], cfg[i + 1], 3, 2, 1)]
            i += 1
        else:
            layers += [NonBottleBlock(cfg[i])]
        i += 1
    
    return layers

def add_upsample(cfg, i, batch_norm=False):
    layers = []
    in_channels = [i, i]

    for k, v in enumerate(cfg):
        if v == 'U':
            layers += [Upsample(in_channels[1], cfg[k + 1][1], stride=2)]
        else:
            layers += [ResidualBlock(v[0], v[1])]
        in_channels = v
    layers += [get_disp(32)]

    return layers

class Net(nn.Module):

    def __init__(self, base, upsample, ringpad=False):
        super(Net, self).__init__()
        self.base = nn.ModuleList(base)
        self.extra = Scene(256, 128)
        self.upsample = nn.ModuleList(upsample)
        self.disp_conv = nn.ModuleList([get_disp(128), get_disp(64), get_disp(16), get_disp(16)])
        self.ringpad = ringpad

    def forward(self, x):
        disp_sources = list()
        disp_skip = list()

        for k in range(len(self.base)):
            ring_pad = isinstance(self.base[k], nn.Conv2d) and self.ringpad
            if ring_pad:
                x = torch.cat([x[..., -1: ], x, x[..., :1]], dim=-1)
            x = self.base[k](x)
            if ring_pad:
                x = x[..., 1: -1]
            if k in [4, 7, 8]:
                disp_skip.append(x)

        if ring_pad:
            x = torch.cat([x[..., -9: ], x, x[..., :9]], dim=-1)
        x = self.extra(x)
        if ring_pad:
            x = x[..., 9: -9]
        
        for k, v in enumerate(self.upsample):
            if k in [2, 4, 6]:
                disp_sources.append(self.disp_conv[k // 2 - 1](x))

            ring_pad = isinstance(v, nn.Conv2d) and self.ringpad
            if ring_pad:
                x = torch.cat([x[..., -1: ], x, x[..., :1]], dim=-1)
            x = v(x)
            if ring_pad:
                x = x[..., 1: -1]

            if k in [0, 2]:
                x = torch.cat([x, disp_skip[1 - k // 2]], dim=1)
        disp_sources.append(x)

        return disp_sources

def build_net(ringpad=False):
    return Net(erfnet(erf_cfg, 3), add_upsample(upsample_cfg, 256), ringpad=False)

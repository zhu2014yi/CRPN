# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and Zhipeng Zhang
# Email: zhangzhipeng2017@ia.ac.cn
# Details: This script provides cross-correlation head of SiamFC
# Reference: SiamFC[] and SiamRPN[Li]
# ------------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
import torch

class Corr_Up(nn.Module):
    """
    SiamFC head
    """
    def __init__(self):
        super(Corr_Up, self).__init__()

    def _conv2d_group(self, x, kernel):
        batch = x.size()[0]
        pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
        px = x.view(1, -1, x.size()[2], x.size()[3])
        po = F.conv2d(px, pk, groups=batch)
        po = po.view(batch, -1, po.size()[2], po.size()[3])
        return po

    def forward(self, z_f, x_f):
        if not self.training:
            return 0.1 * F.conv2d(x_f, z_f)
        else:
            return 0.1 * self._conv2d_group(x_f, z_f)


class RPN_Up(nn.Module):
    """
    For SiamRPN
    """
    def __init__(self, anchor_nums=5, inchannels=256, outchannels=256, cls_type='thinner'):
        super(RPN_Up, self).__init__()

        self.anchor_nums = anchor_nums
        self.inchannels = inchannels
        self.outchannels = outchannels

        if cls_type == 'thinner': self.cls_channel = self.anchor_nums
        elif cls_type == 'thicker': self.cls_channel = self.anchor_nums * 2
        else: raise ValueError('not implemented cls/loss type')

        self.reg_channel = 4 * self.anchor_nums

        self.template_cls = nn.Conv2d(self.inchannels, self.outchannels * self.cls_channel, kernel_size=3)
        self.template_reg = nn.Conv2d(self.inchannels, self.outchannels * self.reg_channel, kernel_size=3)

        self.search_cls = nn.Conv2d(self.inchannels, self.outchannels, kernel_size=3)
        self.search_reg = nn.Conv2d(self.inchannels, self.outchannels, kernel_size=3)
        self.adjust = nn.Conv2d(self.reg_channel, self.reg_channel, kernel_size=1)


    def _conv2d_group(self, x, kernel):
        batch = kernel.size()[0]
        pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
        px = x.view(1, -1, x.size()[2], x.size()[3])
        po = F.conv2d(px, pk, groups=batch)
        po = po.view(batch, -1, po.size()[2], po.size()[3])
        return po


    def forward(self, z_f, x_f):
        cls_kernel = self.template_cls(z_f)
        reg_kernel = self.template_reg(z_f)

        cls_feature = self.search_cls(x_f)
        loc_feature = self.search_reg(x_f)

        _, _, s_cls, _ = cls_kernel.size()
        _, _, s_reg, _ = reg_kernel.size()

        pred_cls = self._conv2d_group(cls_feature, cls_kernel)
        pred_reg = self.adjust(self._conv2d_group(loc_feature, reg_kernel))

        return pred_cls, pred_reg

class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, hidden_kernel_size=5):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1)
        )

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)  # 1,256,5,5
        search = self.conv_search(search)  # 1,256,29,29
        feature = xcorr_depthwise(search, kernel)
        out = self.head(feature)
        return out


class DepthwiseRPN(RPN):
    def __init__(self, anchor_num=5, in_channels=512, out_channels=256):
        super(DepthwiseRPN, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)  # 1,256,7,7  1,256,31,31
        loc = self.loc(z_f, x_f)
        return cls, loc



def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out
class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )
    def forward(self, x):
        x = self.downsample(x)
        return x

class FTB_v1(nn.Module):
    """
    特征转移块

    """
    def __init__(self,C,C_):
        super(FTB_v1,self).__init__()
        self.relu=nn.ReLU(inplace=True)
        self.channel1=nn.Sequential(
            nn.Conv2d(in_channels=C,out_channels=C_,kernel_size=3,bias=False),
            nn.BatchNorm2d(C_)
        )
        self.fuse=nn.Conv2d(in_channels=C_+C_,out_channels=C_,kernel_size=1,bias=False)
    def forward(self, x1,x2):
        #x1_test=x1.detach().cpu().numpy()
        x2=self.channel1(x2)
        #x2_test=x2.detach().cpu().numpy()
        out=torch.cat((x1,x2),dim=1)
        out=self.fuse(out)
        return out

class Adjustlayer_resnet(nn.Module):
    def __init__(self,inchannels,out_channels):
        super(Adjustlayer_resnet,self).__init__()
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels=inchannels,out_channels=out_channels,kernel_size=3,bias=False),
            nn.BatchNorm2d(inchannels)
        )
    def forward(self, x):
        x=self.downsample(x)
        return  x
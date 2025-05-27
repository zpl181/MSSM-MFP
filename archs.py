import time

import numpy as np
import torch
#from numpy.ma.bench import xs
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import seaborn as sns
import os
import matplotlib.pyplot as plt
from utils import *

__all__ = ['MSSM-MFP']

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
from mmcv.cnn import ConvModule
import pdb


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixerBlock(nn.Module):
    def __init__(self, dim=256, depth=7, k=7):
        super(ConvMixerBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(dim, dim, kernel_size=(k, k), groups=dim, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            ) for i in range(depth)]
        )

    def forward(self, x):
        x = self.block(x)
        return x




def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)  # 卷积核大小为 1x1，通常用于降维或增加特征映射的通道数


class ShiftedOperator:
    def __init__(self, pad):
        self.pad = pad
    def shift(self, dim, xs, H, W):
        x_shift = [torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_cat = torch.narrow(x_cat, 3, self.pad, W)
        return x_cat




class ChannelSELayer(torch.nn.Module):
    """
    Implements Squeeze and Excitation
    """

    def __init__(self, num_channels):
        """
        Initialization

        Args:
            num_channels (int): No of input channels
        """

        super(ChannelSELayer, self).__init__()

        self.gp_avg_pool = torch.nn.AdaptiveAvgPool2d(1)

        self.reduction_ratio = 8  # default reduction ratio

        num_channels_reduced = num_channels // self.reduction_ratio

        self.fc1 = torch.nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = torch.nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.act = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm2d(num_channels)


    def forward(self, inp):

        batch_size, num_channels, H, W = inp.size()

        out = self.act(self.fc1(self.gp_avg_pool(inp).view(batch_size, num_channels)))
        out = self.sigmoid(self.fc2(out))

        out = torch.mul(inp, out.view(batch_size, num_channels, 1, 1))

        out = self.bn(out)
        out = self.act(out)

        return out



class ResPath(torch.nn.Module):
    """
    Implements ResPath-like modified skip connection

    """

    def __init__(self, in_chnls, n_lvl):
        """
        Initialization

        Args:
            in_chnls (int): number of input channels
            n_lvl (int): number of blocks or levels
        """

        super(ResPath, self).__init__()

        self.convs = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])
        self.sqes = torch.nn.ModuleList([])

        self.bn = torch.nn.BatchNorm2d(in_chnls)
        self.act = torch.nn.LeakyReLU()
        self.sqe = torch.nn.BatchNorm2d(in_chnls)

        for i in range(n_lvl):
            self.convs.append(
                torch.nn.Conv2d(in_chnls, in_chnls, kernel_size=(3, 3), padding=1)
            )
            self.bns.append(torch.nn.BatchNorm2d(in_chnls))
            self.sqes.append(ChannelSELayer(in_chnls))


    def forward(self, x):

        for i in range(len(self.convs)):
            x = x + self.sqes[i](self.act(self.bns[i](self.convs[i](x))))

        return self.sqe(self.act(self.bn(x)))






class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1, shift_size=5):
        super().__init__()
        out_features = out_features or in_features  # in_features: 输入特征的维度，这是模型接受的输入数据的特征维度。
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size  #  位移的大小，指定了在Shift-MLP中进行位移操作时应该如何操作。
        self.pad = shift_size // 2   # 填充大小，它是位移大小的一半，通常用于在位移操作中处理边界情况。

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    #     def shift(x, dim):
    #         x = F.pad(x, "constant", 0)
    #         x = torch.chunk(x, shift_size, 1)
    #         x = [ torch.roll(x_c, shift, dim) for x_s, shift in zip(x, range(-pad, pad+1))]
    #         x = torch.cat(x, 1)
    #         return x[:, :, pad:-pad, pad:-pad]

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)  # （左边距，右边距，上边距，下边距）
        xs = torch.chunk(xn, self.shift_size, 1)  # 将张量分割成若干块   将张量 xn 沿着第 1 维（通常是通道维度）分割成了 self.shift_size 个块
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]  # 负值表示向左滚动，正值表示向右滚动  通过 zip 函数，将这两个可迭代对象逐一配对，得到一个迭代器，其中每个元素是一个元组，包含了 xs 中的一个分块张量和对应的滚动位移量
        x_cat = torch.cat(x_shift, 1)  # 张量沿着第1维（通道维度）进行拼接
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)

        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)

        x = self.fc1(x_shift_r)

        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)#
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x


class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)  # trunc_normal_ 是一个函数，它以正态分布方式初始化权重，标准差为0.02
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 将偏置初始化为0。
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # fan_out 表示从该层输出的连接数量
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):   # 张量x的高度和宽度
        B, N, C = x.shape  # (B, C, N) 表示 x 的形状
        x = x.transpose(1, 2).view(B, C, H, W)  # 将输入张量 x 的维度重新排列   transpose(1, 2) 操作表示将第 1 维和第 2 维进行交换  view重塑操作 三维重塑成四维
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)  # (B, C, H * W) -> (B, H * W, C)

        return x


class MSAG(nn.Module):
    """
    Multi-scale attention gate
    """
    def __init__(self, channel):
        super(MSAG, self).__init__()
        self.channel = channel
        self.pointwiseConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.ordinaryConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.dilationConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=2, stride=1, dilation=2, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.dilationConv1 = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=8, stride=1, dilation=8, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        # self.dilationConv2 = nn.Sequential(
        #     nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=7, stride=1, dilation=7, bias=True),
        #     nn.BatchNorm2d(self.channel),
        # )
        self.voteConv = nn.Sequential(
            nn.Conv2d(self.channel * 4, self.channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.channel),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.pointwiseConv(x)
        x2 = self.ordinaryConv(x)
        x3 = self.dilationConv(x)
        x4 = self.dilationConv1(x)
        #x5 = self.dilationConv2(x)
        _x = self.relu(torch.cat((x1, x2, x3, x4), dim=1))
        _x = self.voteConv(_x)
        x = x + x * _x
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)  # 用来确保 img_size 变量是一个包含两个元素的元组 (tuple) 的函数
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]  # 图像的高度（self.H）和宽度（self.W）
        self.num_patches = self.H * self.W  # 计算图像块的个数
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class MSSM-MFP(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        #self.stem = nn.Conv2d(input_channels, 6, kernel_size=1)

        #self.encoder0 = nn.Conv2d(1, 16, 3, stride=1, dilation=1, padding=1)
        self.encoder1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        #self.ConvMixer = ConvMixerBlock(dim=256, depth=7, k=7)

        #self.dilate0 = nn.Conv2d(16, 16, kernel_size=3, dilation=1, padding=1)
        self.dilate1 = nn.Conv2d(32, 32, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(64, 64, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(128, 128, kernel_size=3, dilation=5, padding=5)
        # self.dilate4 = nn.Conv2d(160, 160, kernel_size=3, dilation=1, padding=1)
        # self.dilate5 = nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1)

        #self.ebn0 = nn.BatchNorm2d(16)
        self.ebn1 = nn.BatchNorm2d(32)
        self.ebn2 = nn.BatchNorm2d(64)
        self.ebn3 = nn.BatchNorm2d(128)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        #self.decoder6 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(64)
        self.dbn4 = nn.BatchNorm2d(32)
        #self.dbn5 = nn.BatchNorm2d(16)





        self.rspth1 = ResPath(32, 3)
        self.rspth2 = ResPath(64, 2)
        self.rspth3 = ResPath(128, 1)
        # self.rspth4 = ResPath(160, 2)
        # self.rspth5 = ResPath(256, 1)
        # self.rspth6 = ResPath(512, 1)

        self.msag6 = MSAG(512)
        self.msag5 = MSAG(256)
        self.msag4 = MSAG(160)
        self.msag3 = MSAG(128)
        self.msag2 = MSAG(64)
        self.msag1 = MSAG(32)





        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

        #self.soft = nn.Softmax(dim=2)

        self.soft = nn.Sigmoid()

    def forward(self, x):

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        #x = self.stem(x)
        ### Stage 1
        # out = F.gelu(F.max_pool2d(self.ebn0(self.encoder0(x)), 2, 2))
        # t0 = out
        # out = self.dilate0(out)

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t_1 = out

        out = F.gelu(self.dilate1(out))


        t1 = out
        ### Stage 2
        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t_2 = out

        out = F.gelu(self.dilate2(out))

        t2 = out
        ### Stage 3
        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t_3 = out

        out = F.gelu(self.dilate3(out))

        t3 = out

        #out = x, t0, t1 t2 t3

        ### Tokenized MLP Stage
        ### Stage 4

        out, H, W = self.patch_embed3(out)
        #out = self.msag4(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t_4 = out

        t4 = out
        # t3 = self.msag3(t3)
        # t2 = self.msag2(t2)
        # t1 = self.msag1(t1)
        #t0 = self.msag0(t0)

        ### Bottleneck

        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t_5 = out


        t5 = out


        # ### Stage 4

        #R

        t1 = self.rspth1(t1)
        t2 = self.rspth2(t2)
        t3 = self.rspth3(t3)


        #ML

        t5 = self.msag5(t5)
        t4 = self.msag4(t4)
        t3 = self.msag3(t3)
        t2 = self.msag2(t2)
        t1 = self.msag1(t1)



        #t5 = torch.add(t_5, t5)
        out = F.relu(F.interpolate(self.dbn1(self.decoder1(t5)), scale_factor=(2, 2), mode='bilinear'))
        #t4 = torch.add(t_4, t4)
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear'))
        #t3 = torch.add(t_3, t3)
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear'))
        #t2 = torch.add(t_2, t2)
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear'))
        #t1 = torch.add(t_1, t1)
        out = torch.add(out, t1)
        #out = F.relu(F.interpolate(self.dbn5(self.decoder5(out)), scale_factor=(2, 2), mode='bilinear'))
        #out = torch.add(out, t0)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))
        final = self.final(out)
        #final=self.soft(final)
        return final




if __name__ == '__main__':
    model = MSSM-MFP(1).to('cpu')
    input = torch.randn(2, 3, 224, 224).to('cpu')
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

if __name__ == '__main__':
    model = MSSM-MFP(1).to('cpu')
    input = torch.randn(2, 3, 224, 224).to('cpu')
    start_time = time.time()
    output = model(input)
    # print(output.shape)  # 输出应为 [2, 1, 224, 224]
    # model = AttU_Net(in_channels=3, num_classes=2).to('cpu')
    # input = torch.randn(4, 3, 240, 240).to('cpu')
    # 记录推理开始时间
    # 记录推理结束时间
    end_time = time.time()

    # 计算推理时间
    inference_time = end_time - start_time
    print(f'推理时间: {inference_time:.6f}秒')  # 输出推理时间

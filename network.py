import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import json
import pdb
import numpy as np
import torch.nn as nn

import lib
import lib.workspace as ws
from lib.utils import *
import torch.nn.functional as F
from itertools import product
from lib.models.decoder import *
from lib.models.encoder import *
from matplotlib import pyplot as plt


def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(out_dim),
        activation, )


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(out_dim), )


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.InstanceNorm3d(out_dim),
        activation, )



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class MISNeR(nn.Module):

    def __init__(self, device):
        super(MISNeR, self).__init__()
        # accepts 128**3 res input

        self.actvn = nn.ReLU()
        self.device = device

        self.down_1 = conv_block_2_3d(1, 16, self.actvn)
        self.down_2 = conv_block_2_3d(16, 32, self.actvn)
        self.down_3 = conv_block_2_3d(32, 64, self.actvn)
        self.down_4 = conv_block_2_3d(64, 128, self.actvn)
        self.down_5 = conv_block_2_3d(128, 128, self.actvn)

        self.fc_1 = nn.utils.weight_norm(nn.Conv1d(3456, 1728, 1))
        self.fc_2 = nn.utils.weight_norm(nn.Conv1d(3456+1728, 2592, 1))
        self.fc_3 = nn.utils.weight_norm(nn.Conv1d(2592+1728, 2160, 1))
        self.fc_4 = nn.utils.weight_norm(nn.Conv1d(2160+864, 1512, 1))
        self.fc_5 = nn.utils.weight_norm(nn.Conv1d(1512+432, 972, 1))
        self.fc_6 = nn.utils.weight_norm(nn.Conv1d(972+27, 1021, 1))
        self.fc_out_0 = nn.Conv1d(1024, 2048, 1)
        self.fc_out_1 = nn.Conv1d(2048, 4096, 1)
        self.fc_out_2 = nn.Conv1d(4096, 1, 1)

        self.maxpool = nn.MaxPool3d(2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(128, 512)
        self.th = nn.Tanh()

        displacment = 2/127#722
        displacments = []

        for i in [-1,0,1]:
            for j in [-1,0,1]:
                for k in [-1,0,1]:
                    displacments.append([i*displacment,j*displacment,k*displacment])

        self.displacments = torch.Tensor(displacments).cuda(self.device)

    def forward(self, x, p):
        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)  # (B,1,7,num_samples,3)
        feature_0 = F.grid_sample(x, p, align_corners=True)  # out : (B,C (of x), 1,1,sample_num)
        shape_0 = feature_0.shape
        feature_0 = torch.reshape(feature_0,
                                 (shape_0[0], shape_0[1] * shape_0[3], shape_0[4]))

        net = self.down_1(x)
        feature_1 = F.grid_sample(net, p, align_corners=True)  # out : (B,C (of x), 1,1,sample_num)
        shape_1 = feature_1.shape
        feature_1 = torch.reshape(feature_1,
                                 (shape_1[0], shape_1[1] * shape_1[3], shape_1[4]))
        net = self.maxpool(net)

        net = self.down_2(net)
        feature_2 = F.grid_sample(net, p, align_corners=True)  # out : (B,C (of x), 1,1,sample_num)
        shape_2 = feature_2.shape
        feature_2 = torch.reshape(feature_2,
                                 (shape_2[0], shape_2[1] * shape_2[3], shape_2[4]))
        net = self.maxpool(net)

        net = self.down_3(net)
        feature_3 = F.grid_sample(net, p, align_corners=True)  # out : (B,C (of x), 1,1,sample_num)
        shape_3 = feature_3.shape
        feature_3 = torch.reshape(feature_3,
                                 (shape_3[0], shape_3[1] * shape_3[3], shape_3[4]))
        net = self.maxpool(net)

        net = self.down_4(net)
        feature_4 = F.grid_sample(net, p, align_corners=True)
        shape_4 = feature_4.shape
        feature_4 = torch.reshape(feature_4,
                                 (shape_4[0], shape_4[1] * shape_4[3], shape_4[4]))
        net = self.maxpool(net)

        net = self.down_5(net)
        feature_5 = F.grid_sample(net, p, align_corners=True)
        shape_5 = feature_5.shape
        feature_5 = torch.reshape(feature_5,
                                 (shape_5[0], shape_5[1] * shape_5[3], shape_5[4]))

        net = self.avgpool(net)
        vecs = net.view(net.size(0), -1)
        vecs = self.fc(vecs)
        vecs = vecs.view(vecs.shape[0], 1, vecs.shape[1]).repeat(1, 10000, 1).reshape(512, -1).unsqueeze(0)

        vecs = torch.cat((p_features, vecs), dim=1)

        net = self.actvn(self.fc_1(feature_5))
        net = torch.cat((net, feature_4), dim=1)
        net = self.actvn(self.fc_2(net))
        net = torch.cat((net, feature_3), dim=1)
        net = self.actvn(self.fc_3(net))
        net = torch.cat((net, feature_2), dim=1)
        net = self.actvn(self.fc_4(net))
        net = torch.cat((net, feature_1), dim=1)
        net = self.actvn(self.fc_5(net))
        net = torch.cat((net, feature_0), dim=1)
        net = self.actvn(self.fc_6(net))
        net = torch.cat((net, p_features), dim=1)
        net = self.actvn(self.fc_out_0(net))
        net = self.actvn(self.fc_out_1(net))
        net = self.th(self.fc_out_2(net))
        out = net.squeeze(1)
        out = out.transpose(0, 1)

        return out, vecs


class Sampler(nn.Module):
    def __init__(self, bilinear=True):
        super(Sampler, self).__init__()

        self.bilinear = bilinear

        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 128, bilinear)
        self.outc = OutConv(128, 1)

    def forward(self, features):

        x = self.up1(features[3].cuda(1), features[2].cuda(1))
        x = self.up2(x, features[1].cuda(1))
        x = self.up3(x, features[0].cuda(1))
        logits = self.outc(x)
        return logits

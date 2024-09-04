#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import pdb
import numpy as np
from lib.utils import *
from itertools import product

class LearntNeighbourhoodSampling(nn.Module):

    def __init__(self, features_count, step = 0):
        super(LearntNeighbourhoodSampling, self).__init__()
        features_count = 256
        D, H, W = 16, 16, 11
        steps = 4
        self.shape = torch.tensor([W, H, D]).cuda(1).float()

        self.shift = torch.tensor(list(product((-1, 0, 1), repeat=3)))[None].float() * torch.tensor([[[2 ** (
                    steps + 1 - step) / (W), 2 ** (steps + 1 - step) / (H), 2 ** (steps + 1 - step) / (D)]]])[None]
        self.shift = self.shift.cuda(1)

        self.sum_neighbourhood = nn.Conv2d(features_count, features_count, kernel_size=(1, 27), padding=0).cuda(1)

        # torch.nn.init.kaiming_normal_(self.sum_neighbourhood.weight, nonlinearity='relu')
        # torch.nn.init.constant_(self.sum_neighbourhood.bias, 0)
        self.shift_delta = nn.Conv1d(features_count, 27 * 3, kernel_size=(1), padding=0).cuda(1)
        self.shift_delta.weight.data.fill_(0.0)
        self.shift_delta.bias.data.fill_(0.0)

        self.feature_diff_1 = nn.Linear(features_count + 3, features_count)
        self.feature_diff_2 = nn.Linear(features_count, features_count)

        self.feature_center_1 = nn.Linear(features_count + 3, features_count)
        self.feature_center_2 = nn.Linear(features_count, features_count)

    def forward(self, voxel_features, vertices):
        vertices = vertices.cuda(1)
        B, N, _ = vertices.shape
        center = vertices[:, :, None, None]
        features = F.grid_sample(voxel_features, center, mode='bilinear', padding_mode='border', align_corners=True)
        features = features[:, :, :, 0, 0]
        shift_delta = self.shift_delta(features).permute(0, 2, 1).view(B, N, 27, 1, 3)
        shift_delta[:, :, 0, :, :] = shift_delta[:, :, 0, :,
                                     :] * 0  # setting first shift to zero so it samples at the exact point

        # neighbourhood = vertices[:, :, None, None] + self.shift[:, :, :, None] + shift_delta
        neighbourhood = vertices[:, :, None, None] + shift_delta
        features = F.grid_sample(voxel_features, neighbourhood, mode='bilinear', padding_mode='border',
                                 align_corners=True)
        features = features[:, :, :, :, 0]
        features = torch.cat([features, neighbourhood.permute(0, 4, 1, 2, 3)[:, :, :, :, 0]], dim=1)

        features_diff_from_center = features - features[:, :, :, 0][:, :, :,
                                               None]  # 0 is the index of the center cordinate in shifts
        features_diff_from_center = features_diff_from_center.permute([0, 3, 2, 1])
        features_diff_from_center = self.feature_diff_1(features_diff_from_center)
        features_diff_from_center = self.feature_diff_2(features_diff_from_center)
        features_diff_from_center = features_diff_from_center.permute([0, 3, 2, 1])

        features_diff_from_center = self.sum_neighbourhood(features_diff_from_center)[:, :, :, 0].transpose(2, 1)

        center_feautres = features[:, :, :, 13].transpose(2, 1)
        center_feautres = self.feature_center_1(center_feautres)
        center_feautres = self.feature_center_2(center_feautres)

        features = center_feautres + features_diff_from_center
        # features = torch.cat([features, vertices], dim=2)
        return features


class DeepSDF(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
        positional_encoding = True,
        fourier_degree = 1
    ):
        super(DeepSDF, self).__init__()
        # latent_size += 1024

        def make_sequence():
            return []
        if positional_encoding is True:
            dims = [latent_size + 2*fourier_degree*3] + dims + [1]
        else:
            dims = [latent_size + 3] + dims + [1]

        self.positional_encoding = positional_encoding
        self.fourier_degree = fourier_degree
        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()
        self.fc = nn.Linear(256,1024)

    # input: N x (L+3)
    def forward(self, latent, xyz):

        if self.positional_encoding:
            xyz = fourier_transform(xyz, self.fourier_degree)
        # coord = xyz.cuda(1).unsqueeze(0).unsqueeze(2).unsqueeze(2)
        # local = F.grid_sample(cube, coord).squeeze(0).squeeze(2).squeeze(2).permute(1,0)
        # local = self.fc(local)
        # latent = torch.cat([latent, local], dim=1)
        input = torch.cat([latent, xyz], dim=1)


        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x



class DeepSDF3D(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
        positional_encoding = False,
        fourier_degree = 1
    ):
        super(DeepSDF3D, self).__init__()

        def make_sequence():
            return []
        if positional_encoding is True:
            dims = [latent_size + 2*fourier_degree*3] + dims + [1]
        else:
            dims = [latent_size + 3] + dims + [1]

        self.positional_encoding = positional_encoding
        self.fourier_degree = fourier_degree
        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()
        self.fc = nn.Linear(256,1024)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_encoder = nn.Linear(704 , latent_size)

    # input: N x (L+3)
    def forward(self, latent, xyz):

        if self.positional_encoding:
            xyz = fourier_transform(xyz, self.fourier_degree)

        coord = xyz.cuda(1).unsqueeze(0).unsqueeze(2).unsqueeze(2)

        features = []
        for i in range(len(latent)):
            features.append(F.grid_sample(latent[i].cuda(1), coord, align_corners=True).cuda(1).squeeze(0).squeeze(2).squeeze(2))

        latents = features[0]

        for i in range(len(latent)-1):
            latents = torch.cat([latents, features[i+1]], dim=0)

        latents = self.fc_encoder(latents.permute(1,0))
        input = torch.cat([latents, xyz.cuda(1)], dim=1)

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x



class Decoder3D(nn.Module):
    def __init__(self):
        super(Decoder3D, self).__init__()
        self.fc_1 = nn.utils.weight_norm(nn.Conv1d(3456, 1728, 1))
        self.fc_2 = nn.utils.weight_norm(nn.Conv1d(3456 + 1728, 2592, 1))
        self.fc_3 = nn.utils.weight_norm(nn.Conv1d(2592 + 1728, 2160, 1))
        self.fc_4 = nn.utils.weight_norm(nn.Conv1d(2160 + 864, 1512, 1))
        self.fc_5 = nn.utils.weight_norm(nn.Conv1d(1512 + 432, 972, 1))
        self.fc_6 = nn.utils.weight_norm(nn.Conv1d(972 + 27, 1024, 1))
        # self.fc_out_0 = nn.Conv1d(1024, 2048, 1)
        self.fc_out_1 = nn.Conv1d(1024, 1, 1)

        self.maxpool = nn.MaxPool3d(2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(128, 512)
        self.actvn = nn.ReLU()

        displacment = 2 / 127  # 722
        displacments = []

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    displacments.append([i * displacment, j * displacment, k * displacment])

        self.displacments = torch.Tensor(displacments).cuda(1)

    def forward(self, x, p):
        p = p.unsqueeze(1).unsqueeze(1).cuda(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)  # (B,1,7,num_samples,3)
        feature_0 = F.grid_sample(x[0].cuda(1), p)  # out : (B,C (of x), 1,1,sample_num)
        shape_0 = feature_0.shape
        feature_0 = torch.reshape(feature_0,
                                  (shape_0[0], shape_0[1] * shape_0[3], shape_0[4]))

        feature_1 = F.grid_sample(x[1].cuda(1), p)  # out : (B,C (of x), 1,1,sample_num)
        shape_1 = feature_1.shape
        feature_1 = torch.reshape(feature_1,
                                  (shape_1[0], shape_1[1] * shape_1[3], shape_1[4]))

        feature_2 = F.grid_sample(x[2].cuda(1), p)  # out : (B,C (of x), 1,1,sample_num)
        shape_2 = feature_2.shape
        feature_2 = torch.reshape(feature_2,
                                  (shape_2[0], shape_2[1] * shape_2[3], shape_2[4]))

        feature_3 = F.grid_sample(x[3].cuda(1), p)  # out : (B,C (of x), 1,1,sample_num)
        shape_3 = feature_3.shape
        feature_3 = torch.reshape(feature_3,
                                  (shape_3[0], shape_3[1] * shape_3[3], shape_3[4]))

        feature_4 = F.grid_sample(x[4].cuda(1), p)
        shape_4 = feature_4.shape
        feature_4 = torch.reshape(feature_4,
                                  (shape_4[0], shape_4[1] * shape_4[3], shape_4[4]))

        feature_5 = F.grid_sample(x[5].cuda(1), p)
        shape_5 = feature_5.shape
        feature_5 = torch.reshape(feature_5,
                                  (shape_5[0], shape_5[1] * shape_5[3], shape_5[4]))

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
        # net = self.actvn(self.fc_out_0(net))
        net = self.fc_out_1(net)
        out = net.squeeze(1)
        out = out.transpose(0, 1)

        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


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


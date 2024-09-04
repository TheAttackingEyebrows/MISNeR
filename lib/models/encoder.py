import functools
import math
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch
import pdb

class _BatchInstanceNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_BatchInstanceNorm, self).__init__(num_features, eps, momentum, affine)
        self.gate = Parameter(torch.Tensor(num_features))
        self.gate.data.fill_(1)
        setattr(self.gate, 'bin_gate', True)

    def forward(self, input):
        self._check_input_dim(input)

        # Batch norm
        if self.affine:
            bn_w = self.weight * self.gate
        else:
            bn_w = self.gate
        out_bn = F.batch_norm(
            input, self.running_mean, self.running_var, bn_w, self.bias,
            self.training, self.momentum, self.eps)

        # Instance norm
        b, c  = input.size(0), input.size(1)
        if self.affine:
            in_w = self.weight * (1 - self.gate)
        else:
            in_w = 1 - self.gate
        input = input.view(1, b * c, *input.size()[2:])
        out_in = F.batch_norm(
            input, None, None, None, None,
            True, self.momentum, self.eps)
        out_in = out_in.view(b, c, *input.size()[2:])
        out_in.mul_(in_w[None, :, None, None])

        return out_bn + out_in


class BatchInstanceNorm1d(_BatchInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))


class BatchInstanceNorm2d(_BatchInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x33D(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, normlayer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = normlayer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = normlayer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, normlayer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = normlayer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = normlayer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = normlayer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, latent_size, depth, basicblock=False, norm_type="bin", inputchannels=1):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        if basicblock:
            block = BasicBlock
        else:
            block = Bottleneck if depth >=44 else BasicBlock

        if norm_type == 'bn':
            from torch.nn import BatchNorm2d
            self.normlayer = functools.partial(BatchNorm2d, affine=True)
        elif norm_type == 'in':
            from torch.nn import InstanceNorm2d
            self.normlayer = functools.partial(InstanceNorm2d, affine=True)
        elif norm_type == 'bin':
            self.normlayer = functools.partial(BatchInstanceNorm2d, affine=True)
        else:
            raise ValueError('Normalization should be either of type')



        self.inplanes = 32
        self.conv1 = nn.Conv2d(inputchannels, 32, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = self.normlayer(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 32, n)
        self.layer2 = self._make_layer(block, 64, n, stride=2)
        self.layer3 = self._make_layer(block, 128, n, stride=2)
        self.layer4 = self._make_layer(block, 256, n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, latent_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, self.normlayer.func):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.normlayer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, normlayer=self.normlayer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, normlayer=self.normlayer))

        return nn.Sequential(*layers)

    def forward(self, x):
        feature_1 = self.conv1(x)
        feature_1 = self.bn1(feature_1)
        feature_1 = self.relu(feature_1)
        feature_2 = self.layer1(feature_1)
        feature_3 = self.layer2(feature_2)
        feature_4 = self.layer3(feature_3)
        feature_out = self.layer4(feature_4)
        x = self.avgpool(feature_out)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, normlayer=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = conv3x33D(inplanes, planes, stride)
        self.bn1 = normlayer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x33D(planes, planes)
        self.bn2 = normlayer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):

    def __init__(self, latent_size, depth, norm_type="in", inputchannels=1):
        super(ResNet3D, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6


        block = BasicBlock3D


        if norm_type == 'bn':
            from torch.nn import BatchNorm3d
            self.normlayer = functools.partial(BatchNorm3d, affine=True)
        elif norm_type == 'in':
            from torch.nn import InstanceNorm3d
            self.normlayer = functools.partial(InstanceNorm3d, affine=True)
        else:
            raise ValueError('Normalization should be either of type')



        self.inplanes = 64
        self.conv1 = nn.Conv3d(inputchannels, 64, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = self.normlayer(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, n)
        self.layer2 = self._make_layer(block, 128, n, stride=2)
        self.layer3 = self._make_layer(block, 256, n, stride=2)
        self.layer4 = self._make_layer(block, 256, n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(256 * block.expansion, latent_size)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, self.normlayer.func):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.normlayer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, normlayer=self.normlayer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, normlayer=self.normlayer))

        return nn.Sequential(*layers)

    def forward(self, x):
        feature_1 = self.conv1(x)
        feature_1 = self.bn1(feature_1)
        feature_1 = self.relu(feature_1)
        feature_2 = self.layer1(feature_1)
        feature_3 = self.layer2(feature_2)
        feature_4 = self.layer3(feature_3)
        feature_5 = self.layer4(feature_4)
        x = self.avgpool(feature_5)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class MultiLayer3D(nn.Module):

    def __init__(self, latent_size, depth, norm_type="in", inputchannels=1):
        super(MultiLayer3D, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6


        block = BasicBlock3D


        if norm_type == 'bn':
            from torch.nn import BatchNorm3d
            self.normlayer = functools.partial(BatchNorm3d, affine=True)
        elif norm_type == 'in':
            from torch.nn import InstanceNorm3d
            self.normlayer = functools.partial(InstanceNorm3d, affine=True)
        else:
            raise ValueError('Normalization should be either of type')



        self.inplanes = 64
        self.conv1 = nn.Conv3d(inputchannels, 64, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = self.normlayer(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, n)
        self.layer2 = self._make_layer(block, 128, n, stride=2)
        self.layer3 = self._make_layer(block, 256, n, stride=2)
        self.layer4 = self._make_layer(block, 256, n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(256 * block.expansion, latent_size)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, self.normlayer.func):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.normlayer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, normlayer=self.normlayer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, normlayer=self.normlayer))

        return nn.Sequential(*layers)

    def forward(self, x):
        feature_1 = self.conv1(x)
        feature_1 = self.bn1(feature_1)
        feature_1 = self.relu(feature_1)
        feature_2 = self.layer1(feature_1)
        feature_3 = self.layer2(feature_2)
        feature_4 = self.layer3(feature_3)
        feature_5 = self.layer4(feature_4)
        x = [feature_2, feature_3, feature_4, feature_5]
        # x = self.avgpool(feature_5)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x


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


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


# class Encoder3D(nn.Module):
#
#     def __init__(self):
#         super(Encoder3D, self).__init__()
#         # Model type specifies number of layers for CIFAR-10 model
#         self.actvn = nn.ReLU()
#
#         self.down_1 = conv_block_2_3d(1, 16, self.actvn)
#         self.down_2 = conv_block_2_3d(16, 32, self.actvn)
#         self.down_3 = conv_block_2_3d(32, 64, self.actvn)
#         self.down_4 = conv_block_2_3d(64, 128, self.actvn)
#         self.down_5 = conv_block_2_3d(128, 128, self.actvn)
#
#
#         self.maxpool = nn.MaxPool3d(2)
#         self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
#         self.fc = nn.Linear(128, 512)
#
#
#     def forward(self, x):
#         x = x.cuda(0)
#
#         features = []
#         net = self.down_1(x)
#         feature_1 = net
#         features.append(feature_1)
#         net = self.maxpool(net)
#
#         net = self.down_2(net)
#         feature_2 = net
#         features.append(feature_2)
#         net = self.maxpool(net)
#
#         net = self.down_3(net)
#         feature_3 = net
#         features.append(feature_3)
#         net = self.maxpool(net)
#
#         net = self.down_4(net)
#         feature_4 = net
#         features.append(feature_4)
#         net = self.maxpool(net)
#
#         net = self.down_5(net)
#         feature_5 = net
#         features.append(feature_5)
#
#         net = self.avgpool(net)
#         vecs = net.view(net.size(0), -1)
#         vecs = self.fc(vecs)
#
#         return features, vecs


class Encoder3D(nn.Module):

    def __init__(self):
        super(Encoder3D, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        self.in_dim = 1
        self.n_classes = 1
        self.n_channels = 1
        self.num_filters = 16
        activation = nn.LeakyReLU(0.2, inplace=True)

        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_4 = max_pooling_3d()

        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 8, self.num_filters * 8, activation)

        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 16, activation)
        self.up_1 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 16, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 8, activation)
        self.up_2 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)


    def forward(self, x):
        x = x.cuda(0)

        down_1 = self.down_1(x)  # -> [1, 4, 128, 128, 128]
        pool_1 = self.pool_1(down_1)  # -> [1, 4, 64, 64, 64]

        down_2 = self.down_2(pool_1)  # -> [1, 8, 64, 64, 64]
        pool_2 = self.pool_2(down_2)  # -> [1, 8, 32, 32, 32]

        down_3 = self.down_3(pool_2)  # -> [1, 16, 32, 32, 32]
        pool_3 = self.pool_3(down_3)  # -> [1, 16, 16, 16, 16]

        down_4 = self.down_4(pool_3)  # -> [1, 32, 16, 16, 16]
        pool_4 = self.pool_4(down_4)  # -> [1, 32, 8, 8, 8]

        # Bridge
        bridge = self.bridge(pool_4)  # -> [1, 128, 4, 4, 4]

        # Up sampling
        trans_1 = self.trans_1(bridge)  # -> [1, 128, 8, 8, 8]
        concat_1 = torch.cat([trans_1, down_4], dim=1)  # -> [1, 192, 8, 8, 8]
        up_1 = self.up_1(concat_1)  # -> [1, 64, 8, 8, 8]

        trans_2 = self.trans_2(up_1)  # -> [1, 64, 16, 16, 16]
        concat_2 = torch.cat([trans_2, down_3], dim=1)  # -> [1, 96, 16, 16, 16]
        up_2 = self.up_2(concat_2)  # -> [1, 32, 16, 16, 16]

        trans_3 = self.trans_3(up_2)  # -> [1, 32, 32, 32, 32]
        concat_3 = torch.cat([trans_3, down_2], dim=1)  # -> [1, 48, 32, 32, 32]
        up_3 = self.up_3(concat_3)  # -> [1, 16, 32, 32, 32]

        trans_4 = self.trans_4(up_3)  # -> [1, 16, 64, 64, 64]
        concat_4 = torch.cat([trans_4, down_1], dim=1)  # -> [1, 24, 64, 64, 64]
        up_4 = self.up_4(concat_4)  # -> [1, 8, 64, 64, 64]


        return up_4



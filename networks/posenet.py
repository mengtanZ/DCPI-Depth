from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torchvision import transforms
from collections import OrderedDict
from timm.models.layers import trunc_normal_
import torch.nn.functional as F




class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        # loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded = torch.hub.load_state_dict_from_url(models.ResNet18_Weights.IMAGENET1K_V1.url)
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class MotionEncoder_old(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(MotionEncoder_old, self).__init__()

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        # self.trans = nn.Linear(in_features=2, out_features=3)

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        self.motion_encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def compute_div(self, flow):
        _, _, h, w = flow.shape
        flow_padded = F.pad(flow, (1, 1, 1, 1), mode='constant', value=0)

        div_flow = - flow_padded[:, 1:, :h, 1:w + 1] \
                   + flow_padded[:, 1:, 2:, 1:w + 1] \
                   + flow_padded[:, 0:1, 1:h + 1, 2:] \
                   - flow_padded[:, 0:1, 1:h + 1, :w]
        return div_flow

    def compute_cur(self, flow):
        _, _, h, w = flow.shape
        flow_padded = F.pad(flow, (1, 1, 1, 1), mode='constant', value=0)

        cur_flow = - flow_padded[:, 0:1, :h, 1:w + 1] \
                   + flow_padded[:, 0:1, 2:, 1:w + 1] \
                   - flow_padded[:, 1:, 1:h + 1, 2:] \
                   + flow_padded[:, 1:, 1:h + 1, :w]
        return cur_flow

    def normalize_flow(self, flow):
        B, C, H, W = flow.shape
        # h 分量除以 h 的大小
        flow[:, 0, :, :] = flow[:, 0, :, :] / H
        # w 分量除以 w 的大小
        flow[:, 1, :, :] = flow[:, 1, :, :] / W
        return flow

    def forward(self, input_image, flow, motion_type = 'full'):
        # x = self.trans(x)
        self.features = []
        dp = 0
        x = (input_image - 0.45) / 0.225
        flow = self.normalize_flow(flow)
        if motion_type == 'rotational':
            dp = self.compute_cur(flow)
        elif motion_type == 'translational':
            dp = self.compute_div(flow)
        elif motion_type == 'full':
            magnitude = torch.sqrt(flow[:, 0, :, :] ** 2 + flow[:, 1, :, :] ** 2)  # [B, H, W]
            dp = magnitude.unsqueeze(1)
        y = torch.cat((flow, dp), dim=1)  # [B, 3, H, W]

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        y = self.motion_encoder.conv1(y)
        y = self.motion_encoder.bn1(y)
        x = self.encoder.relu(x + y) #-2添加了，-1没添加
        x = self.encoder.layer1(self.encoder.maxpool(x))
        y = self.motion_encoder.layer1(self.encoder.maxpool(y))
        x = x + y
        # self.features.append(x)
        x = self.encoder.layer2(x)
        y = self.motion_encoder.layer2(y)
        x = x + y
        # self.features.append(x)
        x = self.encoder.layer3(x)
        y = self.motion_encoder.layer3(y)
        x = x + y
        # self.features.append(x)
        x = self.encoder.layer4(x)
        y = self.motion_encoder.layer4(y)
        x = x + y
        self.features.append(x)

        return self.features


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class MotionEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(MotionEncoder, self).__init__()

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        # self.trans = nn.Linear(in_features=2, out_features=3)

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        self.motion_encoder = resnets[num_layers](pretrained)
        self.ca = nn.ModuleList([ChannelAttention(planes) for planes in self.num_ch_enc])
        self.sa = nn.ModuleList([SpatialAttention() for _ in self.num_ch_enc])
        # self.sa = [SpatialAttention() for planes in self.num_ch_enc]

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def compute_div(self, flow):
        _, _, h, w = flow.shape
        flow_padded = F.pad(flow, (1, 1, 1, 1), mode='constant', value=0)

        div_flow = - flow_padded[:, 1:, :h, 1:w + 1] \
                   + flow_padded[:, 1:, 2:, 1:w + 1] \
                   + flow_padded[:, 0:1, 1:h + 1, 2:] \
                   - flow_padded[:, 0:1, 1:h + 1, :w]
        return div_flow

    def compute_cur(self, flow):
        _, _, h, w = flow.shape
        flow_padded = F.pad(flow, (1, 1, 1, 1), mode='constant', value=0)

        cur_flow = - flow_padded[:, 0:1, :h, 1:w + 1] \
                   + flow_padded[:, 0:1, 2:, 1:w + 1] \
                   - flow_padded[:, 1:, 1:h + 1, 2:] \
                   + flow_padded[:, 1:, 1:h + 1, :w]
        return cur_flow

    def normalize_flow(self, flow):
        B, C, H, W = flow.shape
        # h 分量除以 h 的大小
        flow[:, 0, :, :] = flow[:, 0, :, :] / H
        # w 分量除以 w 的大小
        flow[:, 1, :, :] = flow[:, 1, :, :] / W
        return flow

    def forward(self, input_image, flow, motion_type = 'full'):
        # x = self.trans(x)
        self.features = []
        dp = 0
        x = (input_image - 0.45) / 0.225
        y = (flow - 0.45) / 0.225

        # flow = self.normalize_flow(flow)
        # if motion_type == 'rotational':
        #     dp = self.compute_cur(flow)
        # elif motion_type == 'translational':
        #     dp = self.compute_div(flow)
        # elif motion_type == 'full':
        #     magnitude = torch.sqrt(flow[:, 0, :, :] ** 2 + flow[:, 1, :, :] ** 2)  # [B, H, W]
        #     dp = magnitude.unsqueeze(1)
        # y = torch.cat((flow, dp), dim=1)  # [B, 3, H, W]

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        y = self.motion_encoder.conv1(y)
        y = self.motion_encoder.bn1(y)
        y = self.sa[0](y) * y
        y = self.ca[0](y) * y
        x = self.encoder.relu(x + y) #-2添加了，-1没添加
        x = self.encoder.layer1(self.encoder.maxpool(x))
        y = self.motion_encoder.layer1(self.encoder.maxpool(y))
        y = self.sa[1](y) * y
        y = self.ca[1](y) * y
        x = x + y
        # self.features.append(x)
        x = self.encoder.layer2(x)
        y = self.motion_encoder.layer2(y)
        y = self.sa[2](y) * y
        y = self.ca[2](y) * y
        x = x + y
        # self.features.append(x)
        x = self.encoder.layer3(x)
        y = self.motion_encoder.layer3(y)
        y = self.sa[3](y) * y
        y = self.ca[3](y) * y
        x = x + y
        # self.features.append(x)
        x = self.encoder.layer4(x)
        y = self.motion_encoder.layer4(y)
        y = self.sa[4](y) * y
        y = self.ca[4](y) * y
        x = x + y
        self.features.append(x)

        return self.features


class Rot_trans_Decoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(Rot_trans_Decoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 3 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        axisangle = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 3)

        return axisangle

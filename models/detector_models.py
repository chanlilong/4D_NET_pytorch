from __future__ import annotations

import math

import numpy as np
import timm
import torch
from torch import nn
from torch.nn import functional as F

from deformable_conv import DeformableConv2d


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1) - torch.log(x2)


def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


def pad_same(x, k: list[int], s: list[int], d: list[int] = (1, 1), value: float = 0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x


def conv2d_same(
        x, weight: torch.Tensor, bias: torch.Tensor | None = None, stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0), dilation: tuple[int, int] = (1, 1), groups: int = 1):
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class resnet_backbone(torch.nn.Module):
    def __init__(self, single_scale=False, dims=256):
        super().__init__()
        self.model = timm.create_model('resnet50', pretrained=True, features_only=True)

        self.conv4 = torch.nn.Conv2d(2048, dims, 1, 1)
        self.conv3 = torch.nn.Conv2d(1024, dims, 1, 1)
        self.conv2 = torch.nn.Conv2d(512, dims, 1, 1)
        self.conv11 = torch.nn.Conv2d(256, dims, 1, 1)
        self.conv00 = torch.nn.Conv2d(64, dims, 1, 1)

    def forward(self, x):
        feats = self.model(x)
        # for f in feats:
        #     print(f.shape)
        #     torch.Size([1, 64, 64, 64])
        #     torch.Size([1, 256, 32, 32])
        #     torch.Size([1, 512, 16, 16])
        #     torch.Size([1, 1024, 8, 8])
        #     torch.Size([1, 2048, 4, 4])

        # must match: torch.Size([1, 256, 32, 32]) torch.Size([1, 512, 16, 16]) torch.Size([1, 1024, 8, 8]) torch.Size([1, 2048, 4, 4])
        l0 = feats[0]
        l1 = feats[1]
        l2 = feats[2]
        l3 = feats[3]
        l4 = feats[4]
        return self.conv00(l0), self.conv11(l1), self.conv2(l2), self.conv3(l3), self.conv4(l4)
        # return self.conv0(l1),self.conv1(F.interpolate(l1,scale_factor=0.75)),self.conv2(l2),self.conv3(l3),self.conv4(l4)


class deform_resnet_backbone(torch.nn.Module):
    def __init__(self, single_scale=False, dims=256):
        super().__init__()
        self.model = timm.create_model('resnet50', pretrained=True, features_only=True)

        self.dconv4 = DeformableConv2d(2048, dims, 1, 1)
        self.dconv3 = DeformableConv2d(1024, dims, 1, 1)
        self.dconv2 = DeformableConv2d(512, dims, 1, 1)
        self.dconv1 = DeformableConv2d(256, dims, 1, 1)
        self.dconv0 = DeformableConv2d(64, dims, 1, 1)

    def forward(self, x):
        feats = self.model(x)
        # for f in feats:
        #     print(f.shape)
        #     torch.Size([1, 64, 64, 64])
        #     torch.Size([1, 256, 32, 32])
        #     torch.Size([1, 512, 16, 16])
        #     torch.Size([1, 1024, 8, 8])
        #     torch.Size([1, 2048, 4, 4])

        # must match: torch.Size([1, 256, 32, 32]) torch.Size([1, 512, 16, 16]) torch.Size([1, 1024, 8, 8]) torch.Size([1, 2048, 4, 4])
        l0 = feats[0]
        l1 = feats[1]
        l2 = feats[2]
        l3 = feats[3]
        l4 = feats[4]
        return self.dconv0(l0), self.dconv1(l1), self.dconv2(l2), self.dconv3(l3), self.dconv4(l4)


class efficientnetv2_s_backbone(torch.nn.Module):
    def __init__(self, dims=256):
        super().__init__()
        self.model = timm.create_model('tf_efficientnetv2_s_in21ft1k', pretrained=True, features_only=True)

        self.conv4 = torch.nn.Conv2d(256, dims, 1, 1)
        self.conv3 = torch.nn.Conv2d(160, dims, 1, 1)
        self.conv2 = torch.nn.Conv2d(64, dims, 1, 1)
        self.conv1 = torch.nn.Conv2d(48, dims, 1, 1)
        self.conv0 = torch.nn.Conv2d(24, dims, 1, 1)

    def forward(self, x):
        feats = self.model(x)

        # torch.Size([1, 24, 256, 256])
        # torch.Size([1, 48, 128, 128])
        # torch.Size([1, 64, 64, 64])
        # torch.Size([1, 160, 32, 32])
        # torch.Size([1, 256, 16, 16])
        l1 = feats[0]
        l2 = feats[1]
        l3 = feats[2]
        l4 = feats[3]
        l5 = feats[4]
        return self.conv0(l1), self.conv1(l2), self.conv2(l3), self.conv3(l4), self.conv4(l5)


class efficientnetv2_s_backbone_old(torch.nn.Module):
    def __init__(self, dims=256):
        super().__init__()
        self.model = timm.create_model('tf_efficientnetv2_s_in21ft1k', pretrained=True, features_only=True)

        self.activation = {}
#         self.layer_names = [f"layer{i+1}" for i in range(4)]
        self.single_scale = single_scale
        blocks = {n: m for n, m in self.model.blocks.named_children()}  # 0,1,2,3,4,5
        if single_scale:
            blocks['5'].register_forward_hook(self.get_activation('layer5'))
            self.conv5 = torch.nn.Conv2d(256, dims, 1, 1)
        else:

            blocks['5'].register_forward_hook(self.get_activation('layer5'))
            blocks['4'].register_forward_hook(self.get_activation('layer4'))
            blocks['3'].register_forward_hook(self.get_activation('layer3'))
            blocks['2'].register_forward_hook(self.get_activation('layer2'))
            blocks['1'].register_forward_hook(self.get_activation('layer1'))

            self.conv5 = torch.nn.Conv2d(256, dims, 1, 1)
            self.conv4 = torch.nn.Conv2d(160, dims, 1, 1)
            self.conv3 = torch.nn.Conv2d(128, dims, 1, 1)
            self.conv2 = torch.nn.Conv2d(64, dims, 1, 1)
            self.conv1 = torch.nn.Conv2d(48, dims, 1, 1)

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output
        return hook

    def forward(self, x):
        _ = self.model.forward_features(x)

        if self.single_scale:
            l5 = self.activation['layer5'].to(x.device)
            self.activation = {}

            return self.conv5(l4)
        else:
            l1 = self.activation['layer1'].to(x.device)
            l2 = self.activation['layer2'].to(x.device)
            l3 = self.activation['layer3'].to(x.device)
            l4 = self.activation['layer4'].to(x.device)
            l5 = self.activation['layer5'].to(x.device)

            self.activation = {}

            return self.conv1(l1), self.conv2(l2), self.conv3(l3), self.conv4(l4), self.conv5(l5)


class BiFPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.num_channels = out_channels

        self.conv7up = nn.Sequential(nn.Conv2d(in_channels, out_channels, (1, 1), bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv6up = nn.Sequential(nn.Conv2d(in_channels, out_channels, (1, 1), bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv5up = nn.Sequential(nn.Conv2d(in_channels, out_channels, (1, 1), bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv4up = nn.Sequential(nn.Conv2d(in_channels, out_channels, (1, 1), bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv3up = nn.Sequential(nn.Conv2d(in_channels, out_channels, (1, 1), bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv4dw = nn.Sequential(nn.Conv2d(in_channels, out_channels, (1, 1), bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv5dw = nn.Sequential(nn.Conv2d(in_channels, out_channels, (1, 1), bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv6dw = nn.Sequential(nn.Conv2d(in_channels, out_channels, (1, 1), bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv7dw = nn.Sequential(nn.Conv2d(in_channels, out_channels, (1, 1), bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, inputs):
        num_channels = self.num_channels
        P3_in, P4_in, P5_in, P6_in, P7_in = inputs  # imgsize: p3: big --> p7: small

        # upsample network
        P7_up = self.conv7up(P7_in)
        P6_up = self.conv6up(P6_in + F.interpolate(P7_up, P6_in.size()[2:], mode='bilinear', align_corners=True))
        P5_up = self.conv5up(P5_in + F.interpolate(P6_up, P5_in.size()[2:], mode='bilinear', align_corners=True))
        P4_up = self.conv4up(P4_in + F.interpolate(P5_up, P4_in.size()[2:], mode='bilinear', align_corners=True))
        P3_out = self.conv3up(P3_in + F.interpolate(P4_up, P3_in.size()[2:], mode='bilinear', align_corners=True))

        # fix to downsample by interpolation
        # downsample networks
        P4_out = self.conv4dw(P4_in + P4_up + F.interpolate(P3_out, P4_up.size()[2:], mode='bilinear', align_corners=True))
        P5_out = self.conv5dw(P5_in + P5_up + F.interpolate(P4_out, P5_up.size()[2:], mode='bilinear', align_corners=True))
        P6_out = self.conv6dw(P6_in + P6_up + F.interpolate(P5_out, P6_up.size()[2:], mode='bilinear', align_corners=True))
        P7_out = self.conv7dw(P7_in + P7_up + F.interpolate(P6_out, P7_up.size()[2:], mode='bilinear', align_corners=True))

        return P3_out, P4_out, P5_out, P6_out, P7_out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Efficient_Det(nn.Module):

    def __init__(self, anchor_dictionary, n_pnts_features=64, n_classes=3, xyz_range=np.array([0, -40.32, -2, 80.64, 40.32, 3])):
        super().__init__()

        self.setup_anchors(anchor_dictionary, xyz_range)
        self.cnn_backbone = resnet_backbone(dims=n_pnts_features)
        self.fpn = BiFPN(n_pnts_features, n_pnts_features)
        self.bb1 = nn.Sequential(nn.Conv2d(n_pnts_features, n_pnts_features // 2, (1, 1), bias=False), nn.BatchNorm2d(n_pnts_features // 2), nn.ReLU(inplace=True), nn.Conv2d(n_pnts_features // 2, 7 * self.n_anchors, (1, 1)))
        self.bb2 = nn.Sequential(nn.Conv2d(n_pnts_features, n_pnts_features // 2, (1, 1), bias=False), nn.BatchNorm2d(n_pnts_features // 2), nn.ReLU(inplace=True), nn.Conv2d(n_pnts_features // 2, 7 * self.n_anchors, (1, 1)))
        self.bb3 = nn.Sequential(nn.Conv2d(n_pnts_features, n_pnts_features // 2, (1, 1), bias=False), nn.BatchNorm2d(n_pnts_features // 2), nn.ReLU(inplace=True), nn.Conv2d(n_pnts_features // 2, 7 * self.n_anchors, (1, 1)))
        self.clss1 = nn.Sequential(nn.Conv2d(n_pnts_features, n_pnts_features // 2, (1, 1), bias=False), nn.BatchNorm2d(n_pnts_features // 2), nn.ReLU(inplace=True), nn.Conv2d(n_pnts_features // 2, n_classes * self.n_anchors, (1, 1)))
        self.clss2 = nn.Sequential(nn.Conv2d(n_pnts_features, n_pnts_features // 2, (1, 1), bias=False), nn.BatchNorm2d(n_pnts_features // 2), nn.ReLU(inplace=True), nn.Conv2d(n_pnts_features // 2, n_classes * self.n_anchors, (1, 1)))
        self.clss3 = nn.Sequential(nn.Conv2d(n_pnts_features, n_pnts_features // 2, (1, 1), bias=False), nn.BatchNorm2d(n_pnts_features // 2), nn.ReLU(inplace=True), nn.Conv2d(n_pnts_features // 2, n_classes * self.n_anchors, (1, 1)))

        self.n_classes = n_classes
        self.n_pnts_features = n_pnts_features

        self.set_cnn_weights()
        # self.return_tensors = return_tensors
        self.xyz_range = xyz_range

    def set_cnn_weights(self):
        weight = self.cnn_backbone.model.conv1.weight.clone()  # 64,3,7,7
        weight = weight.repeat(1, 44, 1, 1)

        self.cnn_backbone.model.conv1 = torch.nn.Conv2d(self.n_pnts_features, 64 * 2, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.cnn_backbone.model.conv1.weight = torch.nn.Parameter(weight[:, :self.n_pnts_features, :, :])
        self.cnn_backbone.model.conv1.requires_grad = True

    def setup_anchors(self, anchor_dictionary, xyz_range):

        self.anchors = anchor_dictionary['anchor_boxes']
        self.anchors.sort(0)
        self.anchors = self.anchors[::-1]
        self.n_anchors = anchor_dictionary['N_anchors']
        self.n_scales = anchor_dictionary['N_scales']

        self.anchors = torch.as_tensor(self.anchors[::-1].copy(), dtype=torch.float32)
        self.anchors = self.anchors.view(self.n_scales, self.n_anchors, 3)

    def forward(self, x):

        fpn_features = self.cnn_backbone(x)

        fpn_features = self.fpn(fpn_features)
        ff1 = fpn_features[2]
        ff2 = fpn_features[3]
        ff3 = fpn_features[4]

        # torch.Size([4, 21, 124, 124])
        # torch.Size([4, 21, 32, 32])
        # torch.Size([4, 21, 16, 16])

        bbox1 = self.bb1(ff1)
        bbox2 = self.bb2(ff2)
        bbox3 = self.bb3(ff3)

        clss1 = self.clss1(ff1)
        clss2 = self.clss2(ff2)
        clss3 = self.clss3(ff3)

        bboxes = []
        classes = []
        for bbox, clss, anchor_i in zip([bbox1, bbox2, bbox3], [clss1, clss2, clss3], self.anchors, strict=False):
            # print(bbox.shape)
            anchor_i = anchor_i.view(1, 1, self.n_anchors, 3).to(bbox.device)
            # anchor_i = anchor_i
            n_batch, C, h, w = bbox.shape
            bbox = bbox.permute(0, 2, 3, 1).contiguous()
            clss = clss.permute(0, 2, 3, 1).contiguous()
            bbox = bbox.view(n_batch, h * w, self.n_anchors, 7)
            clss = clss.view(n_batch, h * w * self.n_anchors, self.n_classes)

            Y, X = torch.meshgrid(torch.linspace(-1, 1, w + 1).type(bbox.type()), torch.linspace(0, 1, h + 1).type(bbox.type()))
            da = torch.sqrt(anchor_i[..., 0:1]**2 + anchor_i[..., 1:2]**2)  # b,1,self.n_anchors,3
            ha = anchor_i[..., 2:3]
            a = (bbox[..., 0:1] * da + (X[:w, :w].flatten() * (self.xyz_range[3] - self.xyz_range[0])).view(1, -1, 1, 1).to(bbox.device)).view(n_batch, -1, 1)  # dx + cx
            b = (bbox[..., 1:2] * da + (Y[:h, :h].flatten() * ((self.xyz_range[4] - self.xyz_range[1]) / 2)).view(1, -1, 1, 1).to(bbox.device)).view(n_batch, -1, 1)  # dy + cy
            c = (bbox[..., 2:3] * ha + ((self.xyz_range[5] - self.xyz_range[2]) / 2)).view(n_batch, -1, 1)  # z
            d = (torch.exp(bbox[..., 3:6]) * (anchor_i.to(bbox.device))).view(n_batch, -1, 3)  # lwh
            e = bbox[..., -1:].view(n_batch, -1, 1).tanh() * np.pi  # r

            bboxes.append(torch.cat([a, b, c, d, e], dim=2))
            classes.append(clss)

        returns = {}
        returns['pred_logits'] = torch.cat(classes, 1)  # b,5249,7
        returns['pred_boxes'] = torch.cat(bboxes, 1)

        return returns

# ruff: noqa: N801, N806
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import timm
import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from torch import Tensor


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1) - torch.log(x2)


def get_same_padding(x: int, k: int, s: int, d: int) -> int:
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


def pad_same(
    x: Tensor,
    k: list[int],
    s: list[int],
    d: list[int] = (1, 1),
    value: float = 0,
) -> Tensor:
    ih, iw = x.size()[-2:]
    pad_h, pad_w = (
        get_same_padding(ih, k[0], s[0], d[0]),
        get_same_padding(iw, k[1], s[1], d[1]),
    )
    if pad_h > 0 or pad_w > 0:
        x = F.pad(
            x,
            [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
            value=value,
        )
    return x


class resnet_backbone(torch.nn.Module):
    def __init__(self, dims: int = 256) -> None:
        super().__init__()
        self.model = timm.create_model('resnet50', pretrained=True, features_only=True)

        self.conv4 = torch.nn.Conv2d(2048, dims, 1, 1)
        self.conv3 = torch.nn.Conv2d(1024, dims, 1, 1)
        self.conv2 = torch.nn.Conv2d(512, dims, 1, 1)
        self.conv11 = torch.nn.Conv2d(256, dims, 1, 1)
        self.conv00 = torch.nn.Conv2d(64, dims, 1, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        feats = self.model(x)

        l0 = feats[0]
        l1 = feats[1]
        l2 = feats[2]
        l3 = feats[3]
        l4 = feats[4]
        return (
            self.conv00(l0),
            self.conv11(l1),
            self.conv2(l2),
            self.conv3(l3),
            self.conv4(l4),
        )


class efficientnetv2_s_backbone(torch.nn.Module):
    def __init__(self, dims: int = 256) -> None:
        super().__init__()
        self.model = timm.create_model(
            'tf_efficientnetv2_s_in21ft1k',
            pretrained=True,
            features_only=True,
        )

        self.conv4 = torch.nn.Conv2d(256, dims, 1, 1)
        self.conv3 = torch.nn.Conv2d(160, dims, 1, 1)
        self.conv2 = torch.nn.Conv2d(64, dims, 1, 1)
        self.conv1 = torch.nn.Conv2d(48, dims, 1, 1)
        self.conv0 = torch.nn.Conv2d(24, dims, 1, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        feats = self.model(x)

        l1 = feats[0]
        l2 = feats[1]
        l3 = feats[2]
        l4 = feats[3]
        l5 = feats[4]
        return (
            self.conv0(l1),
            self.conv1(l2),
            self.conv2(l3),
            self.conv3(l4),
            self.conv4(l5),
        )


class BiFPN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.num_channels = out_channels

        self.conv7up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv6up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv5up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv4up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv3up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv4dw = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv5dw = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv6dw = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv7dw = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        P3_in, P4_in, P5_in, P6_in, P7_in = inputs  # imgsize: p3: big --> p7: small

        # upsample network
        P7_up = self.conv7up(P7_in)
        P6_up = self.conv6up(
            P6_in
            + F.interpolate(
                P7_up,
                P6_in.size()[2:],
                mode='bilinear',
                align_corners=True,
            ),
        )
        P5_up = self.conv5up(
            P5_in
            + F.interpolate(
                P6_up,
                P5_in.size()[2:],
                mode='bilinear',
                align_corners=True,
            ),
        )
        P4_up = self.conv4up(
            P4_in
            + F.interpolate(
                P5_up,
                P4_in.size()[2:],
                mode='bilinear',
                align_corners=True,
            ),
        )
        P3_out = self.conv3up(
            P3_in
            + F.interpolate(
                P4_up,
                P3_in.size()[2:],
                mode='bilinear',
                align_corners=True,
            ),
        )

        # fix to downsample by interpolation
        # downsample networks
        P4_out = self.conv4dw(
            P4_in
            + P4_up
            + F.interpolate(
                P3_out,
                P4_up.size()[2:],
                mode='bilinear',
                align_corners=True,
            ),
        )
        P5_out = self.conv5dw(
            P5_in
            + P5_up
            + F.interpolate(
                P4_out,
                P5_up.size()[2:],
                mode='bilinear',
                align_corners=True,
            ),
        )
        P6_out = self.conv6dw(
            P6_in
            + P6_up
            + F.interpolate(
                P5_out,
                P6_up.size()[2:],
                mode='bilinear',
                align_corners=True,
            ),
        )
        P7_out = self.conv7dw(
            P7_in
            + P7_up
            + F.interpolate(
                P6_out,
                P7_up.size()[2:],
                mode='bilinear',
                align_corners=True,
            ),
        )

        return P3_out, P4_out, P5_out, P6_out, P7_out


class Efficient_Det(nn.Module):
    def __init__(
        self,
        anchor_dictionary: dict[str | np.ndarray],
        n_pnts_features: int = 64,
        n_classes: int = 3,
        xyz_range: np.ndarray | None = None,
    ) -> None:
        if xyz_range is None:
            xyz_range = np.array([0, -40.32, -2, 80.64, 40.32, 3])
        super().__init__()

        self.setup_anchors(anchor_dictionary)
        self.cnn_backbone = resnet_backbone(dims=n_pnts_features)
        self.fpn = BiFPN(n_pnts_features, n_pnts_features)
        self.bb1 = nn.Sequential(
            nn.Conv2d(n_pnts_features, n_pnts_features // 2, (1, 1), bias=False),
            nn.BatchNorm2d(n_pnts_features // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_pnts_features // 2, 7 * self.n_anchors, (1, 1)),
        )
        self.bb2 = nn.Sequential(
            nn.Conv2d(n_pnts_features, n_pnts_features // 2, (1, 1), bias=False),
            nn.BatchNorm2d(n_pnts_features // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_pnts_features // 2, 7 * self.n_anchors, (1, 1)),
        )
        self.bb3 = nn.Sequential(
            nn.Conv2d(n_pnts_features, n_pnts_features // 2, (1, 1), bias=False),
            nn.BatchNorm2d(n_pnts_features // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_pnts_features // 2, 7 * self.n_anchors, (1, 1)),
        )
        self.clss1 = nn.Sequential(
            nn.Conv2d(n_pnts_features, n_pnts_features // 2, (1, 1), bias=False),
            nn.BatchNorm2d(n_pnts_features // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_pnts_features // 2, n_classes * self.n_anchors, (1, 1)),
        )
        self.clss2 = nn.Sequential(
            nn.Conv2d(n_pnts_features, n_pnts_features // 2, (1, 1), bias=False),
            nn.BatchNorm2d(n_pnts_features // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_pnts_features // 2, n_classes * self.n_anchors, (1, 1)),
        )
        self.clss3 = nn.Sequential(
            nn.Conv2d(n_pnts_features, n_pnts_features // 2, (1, 1), bias=False),
            nn.BatchNorm2d(n_pnts_features // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_pnts_features // 2, n_classes * self.n_anchors, (1, 1)),
        )

        self.n_classes = n_classes
        self.n_pnts_features = n_pnts_features

        self.set_cnn_weights()
        # self.return_tensors = return_tensors
        self.xyz_range = xyz_range

    def set_cnn_weights(self) -> None:
        weight = self.cnn_backbone.model.conv1.weight.clone()  # 64,3,7,7
        weight = weight.repeat(1, 44, 1, 1)

        self.cnn_backbone.model.conv1 = torch.nn.Conv2d(
            self.n_pnts_features,
            64 * 2,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.cnn_backbone.model.conv1.weight = torch.nn.Parameter(
            weight[:, : self.n_pnts_features, :, :],
        )
        self.cnn_backbone.model.conv1.requires_grad = True

    def setup_anchors(self, anchor_dictionary: dict[str, np.ndarray]) -> None:
        self.anchors = anchor_dictionary['anchor_boxes']
        self.anchors.sort(0)
        self.anchors = self.anchors[::-1]
        self.n_anchors = anchor_dictionary['N_anchors']
        self.n_scales = anchor_dictionary['N_scales']

        self.anchors = torch.as_tensor(self.anchors[::-1].copy(), dtype=torch.float32)
        self.anchors = self.anchors.view(self.n_scales, self.n_anchors, 3)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
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
        for bbox, clss, anchor_i in zip(
            [bbox1, bbox2, bbox3],
            [clss1, clss2, clss3],
            self.anchors,
            strict=False,
        ):
            # print(bbox.shape)
            anchor_i = anchor_i.view(1, 1, self.n_anchors, 3).to(bbox.device)
            # anchor_i = anchor_i
            n_batch, C, h, w = bbox.shape
            bbox = bbox.permute(0, 2, 3, 1).contiguous()
            clss = clss.permute(0, 2, 3, 1).contiguous()
            bbox = bbox.view(n_batch, h * w, self.n_anchors, 7)
            clss = clss.view(n_batch, h * w * self.n_anchors, self.n_classes)

            Y, X = torch.meshgrid(  # noqa: N806
                torch.linspace(-1, 1, w + 1).type(bbox.type()),
                torch.linspace(0, 1, h + 1).type(bbox.type()),
            )
            da = torch.sqrt(
                anchor_i[..., 0:1] ** 2 + anchor_i[..., 1:2] ** 2,
            )  # b,1,self.n_anchors,3
            ha = anchor_i[..., 2:3]
            x_anchor = X[:w, :w].flatten() * (
                self.xyz_range[3] - self.xyz_range[0]
            ).view(1, -1, 1, 1).to(bbox.device)
            y_anchor = Y[:h, :h].flatten() * (
                (self.xyz_range[4] - self.xyz_range[1]) / 2
            ).view(1, -1, 1, 1).to(bbox.device)
            z_anchor = (self.xyz_range[5] - self.xyz_range[2]) / 2
            a = (bbox[..., 0:1] * da + x_anchor).view(n_batch, -1, 1)  # dx + cx
            b = (bbox[..., 1:2] * da + y_anchor).view(n_batch, -1, 1)  # dy + cy
            c = (bbox[..., 2:3] * ha + z_anchor).view(n_batch, -1, 1)  # z
            d = torch.exp(bbox[..., 3:6]) * (anchor_i.to(bbox.device))  # lwh
            d = d.view(n_batch, -1, 3)
            e = bbox[..., -1:].view(n_batch, -1, 1).tanh() * np.pi  # r

            bboxes.append(torch.cat([a, b, c, d, e], dim=2))
            classes.append(clss)

        returns = {}
        returns['pred_logits'] = torch.cat(classes, 1)  # b,5249,7
        returns['pred_boxes'] = torch.cat(bboxes, 1)

        return returns

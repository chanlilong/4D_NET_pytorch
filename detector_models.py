import timm
import torch

from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from torch.nn import functional as F
import numpy as np
from deformable_conv import DeformableConv2d
from typing import Tuple, Optional ,List  
import math

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1)-torch.log(x2)   

def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)
def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x

def conv2d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
class resnet_backbone(torch.nn.Module):
    def __init__(self,single_scale=False,dims=256):
        super().__init__()
        self.model = timm.create_model('resnet50', pretrained=True, features_only=True)

        self.conv4 = torch.nn.Conv2d(2048,dims,1,1)
        self.conv3 = torch.nn.Conv2d(1024,dims,1,1)
        self.conv2 = torch.nn.Conv2d(512,dims,1,1)
        self.conv11 = torch.nn.Conv2d(256,dims,1,1)
        self.conv00 = torch.nn.Conv2d(64,dims,1,1)
        
    def forward(self,x):
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
        return self.conv00(l0),self.conv11(l1),self.conv2(l2),self.conv3(l3),self.conv4(l4) 
        # return self.conv0(l1),self.conv1(F.interpolate(l1,scale_factor=0.75)),self.conv2(l2),self.conv3(l3),self.conv4(l4)   
        
class deform_resnet_backbone(torch.nn.Module):
    def __init__(self,single_scale=False,dims=256):
        super().__init__()
        self.model = timm.create_model('resnet50', pretrained=True, features_only=True)

        self.dconv4 = DeformableConv2d(2048,dims,1,1)
        self.dconv3 = DeformableConv2d(1024,dims,1,1)
        self.dconv2 = DeformableConv2d(512,dims,1,1)
        self.dconv1 = DeformableConv2d(256,dims,1,1)
        self.dconv0 = DeformableConv2d(64,dims,1,1)
        
    def forward(self,x):
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
        return self.dconv0(l0),self.dconv1(l1),self.dconv2(l2),self.dconv3(l3),self.dconv4(l4) 
        
class efficientnetv2_s_backbone(torch.nn.Module):
    def __init__(self,dims=256):
        super().__init__()
        self.model = timm.create_model('tf_efficientnetv2_s_in21ft1k', pretrained=True, features_only=True)

        self.conv4 = torch.nn.Conv2d(256,dims,1,1)
        self.conv3 = torch.nn.Conv2d(160,dims,1,1)
        self.conv2 = torch.nn.Conv2d(64,dims,1,1)
        self.conv1 = torch.nn.Conv2d(48,dims,1,1)
        self.conv0 = torch.nn.Conv2d(24,dims,1,1)
        
        
    def forward(self,x):
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
        return self.conv0(l1),self.conv1(l2),self.conv2(l3),self.conv3(l4),self.conv4(l5) 

        
class efficientnetv2_s_backbone_old(torch.nn.Module):
    def __init__(self,dims=256):
        super().__init__()
        self.model = timm.create_model('tf_efficientnetv2_s_in21ft1k', pretrained=True, features_only=True)
        
        self.activation = {}
#         self.layer_names = [f"layer{i+1}" for i in range(4)]
        self.single_scale = single_scale
        blocks = {n:m for n,m in self.model.blocks.named_children()} #0,1,2,3,4,5
        if single_scale:
            blocks["5"].register_forward_hook(self.get_activation('layer5'))
            self.conv5 = torch.nn.Conv2d(256,dims,1,1)
        else:

            blocks["5"].register_forward_hook(self.get_activation('layer5'))
            blocks["4"].register_forward_hook(self.get_activation('layer4'))
            blocks["3"].register_forward_hook(self.get_activation('layer3'))
            blocks["2"].register_forward_hook(self.get_activation('layer2'))
            blocks["1"].register_forward_hook(self.get_activation('layer1'))
            
            self.conv5 = torch.nn.Conv2d(256,dims,1,1)
            self.conv4 = torch.nn.Conv2d(160,dims,1,1)
            self.conv3 = torch.nn.Conv2d(128,dims,1,1)
            self.conv2 = torch.nn.Conv2d(64,dims,1,1)
            self.conv1 = torch.nn.Conv2d(48,dims,1,1)
        
    def get_activation(self,name):
        def hook(model, input, output):
            self.activation[name] = output
        return hook
        
    def forward(self,x):
        _ = self.model.forward_features(x)
        
        if self.single_scale:
            l5 = self.activation["layer5"].to(x.device)
            self.activation = {}

            return self.conv5(l4)
        else:
            l1 = self.activation["layer1"].to(x.device)
            l2 = self.activation["layer2"].to(x.device) 
            l3 = self.activation["layer3"].to(x.device)
            l4 = self.activation["layer4"].to(x.device) 
            l5 = self.activation["layer5"].to(x.device)
        
            self.activation = {}

            return self.conv1(l1),self.conv2(l2),self.conv3(l3),self.conv4(l4),self.conv5(l5)

# from torch_dwconv import depthwise_conv2d, DepthwiseConv2d

# class DepthwiseConvBlock(nn.Module):
#     """
#     Depthwise seperable convolution. 
    
    
#     """
#     def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
#         super(DepthwiseConvBlock,self).__init__()
# #         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
# #                                padding, dilation, groups=in_channels, bias=False)
        
# #         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
# #                                    stride=1, padding=0, dilation=1, groups=1, bias=False)
        
#         self.dw_conv = DepthwiseConv2d(in_channels, out_channels,kernel_size,stride,padding,dilation)
        
#         self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
#         self.act = nn.ReLU(inplace=True)
        
#     def forward(self, inputs):
#         # x = self.depthwise(inputs)
#         # x = self.pointwise(x)
        
#         x = self.dw_conv(inputs)
#         x = self.bn(x)
#         return self.act(x)
    
# class DepthwiseConvBlock(nn.Module):
#     """
#     Depthwise seperable convolution. 
    
    
#     """
#     def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
#         super(DepthwiseConvBlock,self).__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
#                                padding, dilation, groups=in_channels, bias=False)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
#                                    stride=1, padding=0, dilation=1, groups=1, bias=False)
        
        
#         self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
#         self.act = nn.LeakyReLU(inplace=True)
        
#     def forward(self, inputs):
#         x = self.depthwise(inputs)
#         x = self.pointwise(x)
#         x = self.bn(x)
#         return self.act(x)
        
        
class BiFPN(nn.Module):
    def __init__(self, in_channels ,out_channels):
        super(BiFPN, self).__init__()
        self.num_channels = out_channels

        # self.conv7up = DepthwiseConvBlock(in_channels,out_channels)
        # self.conv6up = DepthwiseConvBlock(in_channels,out_channels)
        # self.conv5up = DepthwiseConvBlock(in_channels,out_channels)
        # self.conv4up = DepthwiseConvBlock(in_channels,out_channels)
        # self.conv3up = DepthwiseConvBlock(in_channels,out_channels)
        # self.conv4dw = DepthwiseConvBlock(in_channels,out_channels)
        # self.conv5dw = DepthwiseConvBlock(in_channels,out_channels)
        # self.conv6dw = DepthwiseConvBlock(in_channels,out_channels)
        # self.conv7dw = DepthwiseConvBlock(in_channels,out_channels)
        self.conv7up = nn.Sequential(nn.Conv2d(in_channels,out_channels,(1,1),bias=False),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
        self.conv6up = nn.Sequential(nn.Conv2d(in_channels,out_channels,(1,1),bias=False),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
        self.conv5up = nn.Sequential(nn.Conv2d(in_channels,out_channels,(1,1),bias=False),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
        self.conv4up = nn.Sequential(nn.Conv2d(in_channels,out_channels,(1,1),bias=False),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
        self.conv3up = nn.Sequential(nn.Conv2d(in_channels,out_channels,(1,1),bias=False),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
        self.conv4dw = nn.Sequential(nn.Conv2d(in_channels,out_channels,(1,1),bias=False),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
        self.conv5dw = nn.Sequential(nn.Conv2d(in_channels,out_channels,(1,1),bias=False),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
        self.conv6dw = nn.Sequential(nn.Conv2d(in_channels,out_channels,(1,1),bias=False),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
        self.conv7dw = nn.Sequential(nn.Conv2d(in_channels,out_channels,(1,1),bias=False),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
        
        
    def forward(self, inputs):
        num_channels = self.num_channels
        P3_in, P4_in, P5_in, P6_in, P7_in = inputs #imgsize: p3: big --> p7: small

        # upsample network
        P7_up = self.conv7up(P7_in)
        P6_up = self.conv6up(P6_in+F.interpolate(P7_up, P6_in.size()[2:],mode = "bilinear",align_corners=True))
        P5_up = self.conv5up(P5_in+F.interpolate(P6_up, P5_in.size()[2:],mode = "bilinear",align_corners=True))
        P4_up = self.conv4up(P4_in+F.interpolate(P5_up, P4_in.size()[2:],mode = "bilinear",align_corners=True))
        P3_out = self.conv3up(P3_in+F.interpolate(P4_up, P3_in.size()[2:],mode = "bilinear",align_corners=True))

        # fix to downsample by interpolation
        # downsample networks
        P4_out = self.conv4dw(P4_in + P4_up+F.interpolate(P3_out, P4_up.size()[2:],mode = "bilinear",align_corners=True))
        P5_out = self.conv5dw(P5_in + P5_up+F.interpolate(P4_out, P5_up.size()[2:],mode = "bilinear",align_corners=True))
        P6_out = self.conv6dw(P6_in + P6_up+F.interpolate(P5_out, P6_up.size()[2:],mode = "bilinear",align_corners=True))
        P7_out = self.conv7dw(P7_in + P7_up+F.interpolate(P6_out, P7_up.size()[2:],mode = "bilinear",align_corners=True))

        
        return P3_out, P4_out, P5_out,P6_out, P7_out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

    

class Efficient_Det(nn.Module):

    
    def __init__(self,anchor_dictionary,n_pnts_features=64,n_classes=3,xyz_range = np.array([0,-40.32,-2,80.64,40.32,3])):
        super().__init__()
        
        #clusters_dic = {"anchor_boxes":cluster_centers,"N_anchors":N_anchors,"N_scales":N_scales}

        self.setup_anchors(anchor_dictionary,xyz_range)
        self.cnn_backbone = resnet_backbone(dims=n_pnts_features)
        # self.cnn_backbone = efficientnetv2_s_backbone(dims=n_pnts_features)
        # self.fpn = nn.Sequential(BiFPN(n_pnts_features,n_pnts_features),BiFPN(n_pnts_features,n_pnts_features),BiFPN(n_pnts_features,n_pnts_features))
        self.fpn = BiFPN(n_pnts_features,n_pnts_features)
        self.bb1 = nn.Sequential(nn.Conv2d(n_pnts_features,n_pnts_features//2,(1,1),bias=False),nn.BatchNorm2d(n_pnts_features//2),nn.ReLU(inplace=True),nn.Conv2d(n_pnts_features//2,7*self.n_anchors,(1,1)))
        self.bb2 = nn.Sequential(nn.Conv2d(n_pnts_features,n_pnts_features//2,(1,1),bias=False),nn.BatchNorm2d(n_pnts_features//2),nn.ReLU(inplace=True),nn.Conv2d(n_pnts_features//2,7*self.n_anchors,(1,1)))
        self.bb3 = nn.Sequential(nn.Conv2d(n_pnts_features,n_pnts_features//2,(1,1),bias=False),nn.BatchNorm2d(n_pnts_features//2),nn.ReLU(inplace=True),nn.Conv2d(n_pnts_features//2,7*self.n_anchors,(1,1)))
        self.clss1 = nn.Sequential(nn.Conv2d(n_pnts_features,n_pnts_features//2,(1,1),bias=False),nn.BatchNorm2d(n_pnts_features//2),nn.ReLU(inplace=True),nn.Conv2d(n_pnts_features//2,n_classes*self.n_anchors,(1,1)))
        self.clss2 = nn.Sequential(nn.Conv2d(n_pnts_features,n_pnts_features//2,(1,1),bias=False),nn.BatchNorm2d(n_pnts_features//2),nn.ReLU(inplace=True),nn.Conv2d(n_pnts_features//2,n_classes*self.n_anchors,(1,1)))
        self.clss3 = nn.Sequential(nn.Conv2d(n_pnts_features,n_pnts_features//2,(1,1),bias=False),nn.BatchNorm2d(n_pnts_features//2),nn.ReLU(inplace=True),nn.Conv2d(n_pnts_features//2,n_classes*self.n_anchors,(1,1)))

        self.n_classes=n_classes
        self.n_pnts_features = n_pnts_features
        
        self.set_cnn_weights()
        # self.return_tensors = return_tensors
        self.xyz_range = xyz_range

    def set_cnn_weights(self):
        weight = self.cnn_backbone.model.conv1.weight.clone() # 64,3,7,7
        weight = weight.repeat(1,44,1,1)

        self.cnn_backbone.model.conv1 = torch.nn.Conv2d(self.n_pnts_features,64*2,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        self.cnn_backbone.model.conv1.weight = torch.nn.Parameter(weight[:,:self.n_pnts_features,:,:])
        self.cnn_backbone.model.conv1.requires_grad = True

#     def set_cnn_weights(self):
#         '''for effnetv2'''
#         weight = self.cnn_backbone.model.conv_stem.weight.clone() # 24,3,7,7
#         weight = weight.repeat(1,44,1,1)

#         self.cnn_backbone.model.conv_stem = Conv2dSame(self.n_pnts_features,64*2,kernel_size=(3,3),stride=(2,2),padding="same",bias=False)
#         self.cnn_backbone.model.conv_stem.weight = torch.nn.Parameter(weight[:,:self.n_pnts_features,:,:])
#         self.cnn_backbone.model.conv_stem.requires_grad = True


    def setup_anchors(self,anchor_dictionary,xyz_range):
        
        # range_l = (xyz_range[3]-xyz_range[0]) #l
        # range_w = (xyz_range[4]-xyz_range[1]) #w
        # range_h = (xyz_range[5]-xyz_range[2]) #h
        # self.anchors = anchor_dictionary["anchor_boxes"]/np.array([[range_l,range_w,range_h]])
        self.anchors = anchor_dictionary["anchor_boxes"]
        # print(self.anchors.shape)
        # print(self.anchors)
        self.anchors.sort(0)
        self.anchors = self.anchors[::-1]
        self.n_anchors = anchor_dictionary["N_anchors"]
        self.n_scales = anchor_dictionary["N_scales"]
        
        self.anchors = torch.as_tensor(self.anchors[::-1].copy(),dtype=torch.float32)
        self.anchors = self.anchors.view(self.n_scales,self.n_anchors,3)
        # self.anchors = torch.nn.Parameter(self.anchors, requires_grad=False)
        # print(self.anchors)
    def forward(self,x):
        
        fpn_features = self.cnn_backbone(x)

        fpn_features = self.fpn(fpn_features)
        ff1 = fpn_features[2]
        ff2 = fpn_features[3]
        ff3 = fpn_features[4]
        
        # torch.Size([4, 21, 124, 124])
        # torch.Size([4, 21, 32, 32])
        # torch.Size([4, 21, 16, 16])
        
        # boxes = []
        # logits = []
        
        # b,C,h,w = ff1.shape
        bbox1 = self.bb1(ff1)
        bbox2 = self.bb2(ff2)
        bbox3 = self.bb3(ff3)
        
        clss1 = self.clss1(ff1)
        clss2 = self.clss2(ff2)
        clss3 = self.clss3(ff3)

        bboxes = []
        classes = []
        for bbox,clss,anchor_i in zip([bbox1,bbox2,bbox3],[clss1,clss2,clss3],self.anchors):
            # print(bbox.shape)
            anchor_i = anchor_i.view(1,1,self.n_anchors,3).to(bbox.device)
            # anchor_i = anchor_i
            n_batch,C,h,w = bbox.shape
            bbox = bbox.permute(0,2,3,1).contiguous()
            clss = clss.permute(0,2,3,1).contiguous()
            bbox = bbox.view(n_batch,h*w,self.n_anchors,7)
            clss = clss.view(n_batch,h*w*self.n_anchors,self.n_classes)

            # Y,X = torch.meshgrid(torch.linspace(0,1,w+1).type(bbox.type()),torch.linspace(0,1,h+1).type(bbox.type()))
            
            # a = (bbox[...,0:1] + inverse_sigmoid(X[:w,:w].flatten()).view(1,-1,1,1).to(bbox.device)).view(n_batch,-1,1).sigmoid() #dx + cx
            # b = (bbox[...,1:2] + inverse_sigmoid(Y[:h,:h].flatten()).view(1,-1,1,1).to(bbox.device)).view(n_batch,-1,1).sigmoid() #dy + cy 
            # c = bbox[...,2:3].view(n_batch,-1,1).sigmoid() #z
            # d = (bbox[...,3:6] + inverse_sigmoid(anchor_i).to(bbox.device)).sigmoid().view(n_batch,-1,3) # lwh
            # e = bbox[...,-1:].view(n_batch,-1,1).tanh()*np.pi #r
            Y,X = torch.meshgrid(torch.linspace(-1,1,w+1).type(bbox.type()),torch.linspace(0,1,h+1).type(bbox.type()))
            da = torch.sqrt(anchor_i[...,0:1]**2+anchor_i[...,1:2]**2)#b,1,self.n_anchors,3
            ha = anchor_i[...,2:3]
            a = (bbox[...,0:1]*da + (X[:w,:w].flatten()*(self.xyz_range[3]-self.xyz_range[0])).view(1,-1,1,1).to(bbox.device)).view(n_batch,-1,1) #dx + cx
            b = (bbox[...,1:2]*da + (Y[:h,:h].flatten()*((self.xyz_range[4]-self.xyz_range[1])/2)).view(1,-1,1,1).to(bbox.device)).view(n_batch,-1,1) #dy + cy 
            c = (bbox[...,2:3]*ha + ((self.xyz_range[5]-self.xyz_range[2])/2)).view(n_batch,-1,1) #z
            d = (torch.exp(bbox[...,3:6])*(anchor_i.to(bbox.device))).view(n_batch,-1,3) # lwh
            e = bbox[...,-1:].view(n_batch,-1,1).tanh()*np.pi #r
            # print(a.shape,b.shape,c.shape,d.shape,e.shape)
            # print(a,b)
            
            bboxes.append(torch.cat([a,b,c,d,e],dim=2))
            classes.append(clss)
            
        returns = {}
        returns["pred_logits"] = torch.cat(classes,1) #b,5249,7
        returns["pred_boxes"] = torch.cat(bboxes,1)
        # print(returns["pred_boxes"][...,0].min(),returns["pred_boxes"][...,0].max())
        # print(returns["pred_boxes"][...,1].min(),returns["pred_boxes"][...,1].max())
        return returns

    
class Efficient_Det_old(nn.Module):
    
    def __init__(self,anchor_dictionary,dims=128,n_pnts_features=128,n_classes=3,xyz_range = np.array([0,-40.32,-2,80.64,40.32,3])):
        super().__init__()
        
        #clusters_dic = {"anchor_boxes":cluster_centers,"N_anchors":N_anchors,"N_scales":N_scales}

        self.setup_anchors(anchor_dictionary,xyz_range)
        # self.cnn_backbone = efficientnetv2_s_backbone()
        self.cnn_backbone = resnet_backbone(dims=dims)
        self.fpn = nn.Sequential(BiFPN(dims,dims),BiFPN(dims,dims),BiFPN(dims,dims))
        self.bbox_layer = nn.ModuleList([nn.Sequential(nn.Linear(dims,64),nn.LeakyReLU(inplace=True),nn.LayerNorm(64),nn.Linear(64,7*self.n_anchors)) for _ in range(self.n_scales)])
        self.clss_layer = nn.ModuleList([nn.Sequential(nn.Linear(dims,64),nn.LeakyReLU(inplace=True),nn.LayerNorm(64),nn.Linear(64,n_classes*self.n_anchors)) for _ in range(self.n_scales)])
        self.n_classes=n_classes
        self.n_pnts_features = n_pnts_features
        
        self.set_cnn_weights()

    def set_cnn_weights(self):
        weight = self.cnn_backbone.model.conv1.weight.clone() # 64,3,7,7
        weight = weight.repeat(1,44,1,1)

        self.cnn_backbone.model.conv1 = torch.nn.Conv2d(self.n_pnts_features,64*2,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        self.cnn_backbone.model.conv1.weight = torch.nn.Parameter(weight[:,:self.n_pnts_features,:,:])
        self.cnn_backbone.model.conv1.requires_grad = True

    def setup_anchors(self,anchor_dictionary,xyz_range):
        
        range_l = (xyz_range[3]-xyz_range[0]) #l
        range_w = (xyz_range[4]-xyz_range[1]) #w
        range_h = (xyz_range[5]-xyz_range[2]) #h
        self.anchors = anchor_dictionary["anchor_boxes"]/np.array([[range_l,range_w,range_h]])
        self.anchors.sort()
        
        
        self.n_anchors = anchor_dictionary["N_anchors"]
        self.n_scales = anchor_dictionary["N_scales"]
        
        self.anchors = torch.as_tensor(self.anchors[::-1].copy(),dtype=torch.float32)
        self.anchors = self.anchors.view(self.n_scales,self.n_anchors,3)
        # self.anchors = torch.nn.parameter.Parameter(self.anchors, requires_grad=True).float()
        # print(self.anchors)
    def forward(self,x):
        
        scale_features = self.cnn_backbone(x)
        fpn_features = self.fpn(scale_features)
        # for f in fpn_features:print(f.shape)
        fpn_features = [fpn_features[i] for i in [2,3,4]]
        # fpn_features = scale_features
        
        # fpn_features = [SE(fpn_features[i]) for i,SE in enumerate(self.SE_nets)]
            

        # preds = []
        
        boxes = []
        logits = []
        for f,anchor_i,bbox_layer,clss_layer in zip(fpn_features,self.anchors,self.bbox_layer,self.clss_layer):
            # pred = layer(f)
            # print(f.shape)
            # torch.Size([2, 256, 126, 126])
            # torch.Size([2, 256, 94, 94])
            # torch.Size([2, 256, 63, 63])
            # torch.Size([2, 256, 32, 32])
            # torch.Size([2, 256, 16, 16])
            b,C,h,w = f.shape
            
            X,Y = torch.meshgrid(torch.linspace(0,1,w+1),torch.linspace(0,1,h+1))
            
            bbox = bbox_layer(f.permute(0,2,3,1).flatten(1,2)).view(b,h*w,self.n_anchors,7) #b,c,h,w ->b,h,w,c ->b,hw,c -> b,hw,anchors,4
            bbox_final = torch.zeros_like(bbox).type(f.type())
            # print(bbox[...,2:4].shape)# torch.Size([1, 144, 3, 2])
            # anchor_i = torch.as_tensor(anchor_i.copy(),dtype=torch.float32).to(x.device).view(1,1,self.n_anchors,2)#anchor,2
            anchor_i = anchor_i.view(1,1,self.n_anchors,3).type(f.type())
            
            
            bbox_final[...,0:1] = bbox[...,0:1].sigmoid() + Y[:w,:w].flatten().view(1,-1,1,1).type(f.type()).to(x.device)
            bbox_final[...,1:2] = bbox[...,1:2].sigmoid() + X[:h,:h].flatten().view(1,-1,1,1).type(f.type()).to(x.device) 
            bbox_final[...,2:3] = bbox[...,2:3].sigmoid() 
            bbox_final[...,3:6] = (bbox[...,3:6] + inverse_sigmoid(anchor_i)).sigmoid()
            
            bbox_final[...,-1:] = bbox[...,-1:].sigmoid()

            boxes.append(bbox_final.flatten(1,2))
            
            
            logit = clss_layer(f.permute(0,2,3,1).flatten(1,2)).view(b,h*w,self.n_anchors,self.n_classes)
            logits.append(logit.flatten(1,2))
            

        returns = {}
        returns["pred_logits"] = torch.cat(logits,1) #b,5249,4
        returns["pred_boxes"] = torch.clamp(torch.cat(boxes,1),0,1)
        
        # topk = returns["pred_logits"].topk(100,1).indices
        # print(topk)
        # returns["pred_logits"] = returns["pred_logits"][topk]
        # returns["pred_boxes"] = returns["pred_boxes"][:,topk,:]
        
        return returns
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
# from util.misc import nested_tensor_from_tensor_list,NestedTensor
# from models import build_model
from detector_models import Efficient_Det,resnet_backbone,BiFPN,SELayer,deform_resnet_backbone
from torch.nn.functional import grid_sample

class Pointnet_Resblock(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.l1 = torch.nn.Conv2d(in_channels,out_channels,(1,1))
        self.l2 = torch.nn.Conv2d(out_channels,out_channels,(1,1))
        self.lx = torch.nn.Conv2d(in_channels,out_channels,(1,1))

        self.norm1 = torch.nn.BatchNorm2d((out_channels))
        self.norm2 = torch.nn.BatchNorm2d((out_channels))
        self.normx = torch.nn.BatchNorm2d((out_channels))
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.relu2 = torch.nn.ReLU(inplace=True)

    def forward(self,pillars):

            x = self.l1(pillars)
            x = self.norm1(x)
            x = self.relu1(x)
            x = self.l2(x)
            x = self.norm2(x)
            x = self.normx(self.lx(pillars))+x
            x = self.relu2(x)
            

            return x
        

class Pillar_Network_SECOND(nn.Module):

    def __init__(self,input_features = 9, n_features=64,n_pnts_pillar=64):
        super().__init__()
        self.n_features= n_features
        self.conv1 = Pointnet_Resblock(input_features,n_features//4)
        self.conv2 = Pointnet_Resblock(n_features//4,n_features//2)
        self.conv3 = Pointnet_Resblock(n_features//2,n_features)
        self.norm2 = torch.nn.BatchNorm2d((n_features))
        self.maxpool = nn.MaxPool2d((1,n_pnts_pillar))

    def forward(self,x):


        x = self.conv1(x)#torch.Size([1, 64, 2060,64])
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.norm2(x)
        x = self.maxpool(x)

        return x
    
class Pseudo_IMG_Scatter(torch.nn.Module):
    def __init__(self,xsize,ysize):
        super().__init__()
        self.xsize = xsize
        self.ysize = ysize
        
    def forward(self,pillars,coord,contains_pillars):
        batch,n_pillars,n_features = pillars.shape #torch.Size([4, 16384, 64])
        masks = (contains_pillars==1).bool().view(batch,-1)
        filtered_outs = pillars[masks.view(batch,-1)]
        filtered_coors = coord[masks.view(batch,-1)]
        pseudo_img = torch.zeros(batch,self.xsize * self.ysize ,n_features).to(pillars.device).type(pillars.type())
        interval_temp = contains_pillars.sum(1).cumsum(0)
        interval = torch.zeros(contains_pillars.shape[0]+1).to(pillars.device).type(pillars.type())
        interval[1:] = interval_temp
        interval = interval.type(torch.int64)
        
        for batch_idx,idx1,idx2 in zip(np.arange(batch),interval[:-1],interval[1:]):
            this_coords = filtered_coors[idx1:idx2].type(torch.int64)
            indices = this_coords[:,1]*self.xsize +this_coords[:,2]
            pseudo_img[batch_idx][indices.type(torch.int64),:] += filtered_outs[idx1:idx2].type(pillars.type())
            
        return pseudo_img.view(batch,self.xsize,self.ysize,n_features).permute(0,3,1,2)
    
    
class Pillars_to_PseudoIMG(torch.nn.Module):
    """
    Converts Pillars to Pseudo Image
    
    Args:
        outs: Output of Neural Network(Pillar_Network_SECOND) shape-> n_batch,n_max_pillars,n_features
        coords: Tensor with shape [n_batch,3(zxy)] to indicate where the pillars are located
        contains_pillars: 1D Tensor that Contains 1 or 0 to indicate if pillar is empty(0) or not
    """
    def __init__(self,size):
        super().__init__()
        self.xsize,self.ysize = size
        self.xsize,self.ysize = int(self.xsize)+1,int(self.ysize)+1 
        
    def forward(self,pillars,coord,contains_pillars):
        batch,n_pillars,n_features = pillars.shape #torch.Size([4, 16384, 64])
        masks = (contains_pillars==1).bool().squeeze(-1)
        filtered_outs = pillars[masks]
        filtered_coors = coord[masks]
        pseudo_img = torch.zeros(batch,self.xsize*self.ysize,n_features).to(pillars.device)
        interval_temp = contains_pillars.sum(1).cumsum(0)
        interval = torch.zeros(contains_pillars.shape[0]+1).to(pillars.device)
        interval[1:] = interval_temp
        interval = interval.long()
        
        for batch_idx,idx1,idx2 in zip(np.arange(batch),interval[:-1],interval[1:]):
            this_coords = filtered_coors[idx1:idx2].long()
            indices = this_coords[:,1]*self.xsize +this_coords[:,2]
            pseudo_img[batch_idx][indices.long(),:] += filtered_outs[idx1:idx2].float()
            
        return pseudo_img.view(batch,self.xsize,self.ysize,n_features).permute(0,3,1,2)
    
class Pseudo_IMG_Scatter_Pillar(torch.nn.Module):
    def __init__(self,xsize,ysize):
        super().__init__()
        self.xsize = xsize
        self.ysize = ysize
        self.xsize,self.ysize = int(self.xsize)+1,int(self.ysize)+1 
        self.dynamic_layer = torch.nn.Linear(64,3) #3 scales
        # self.dynamic_se_layer = SELayer(3, reduction=3)
        
    def forward(self,pillars,coord,contains_pillars):
        batch,n_pillars,n_features = pillars.shape #torch.Size([4, 16384, 64])
        masks = (contains_pillars==1).bool().view(batch,-1)
        filtered_outs = pillars[masks.view(batch,-1)]# torch.Size([11944, 64])
        filtered_coors = coord[masks.view(batch,-1)]
        pseudo_img = torch.zeros(batch,self.xsize * self.ysize ,n_features).to(pillars.device).type(pillars.type())
        dynamic_img = torch.zeros(batch,self.xsize * self.ysize ,3).to(pillars.device).type(pillars.type())
        interval_temp = contains_pillars.sum(1).cumsum(0)
        interval = torch.zeros(contains_pillars.shape[0]+1).to(pillars.device).type(pillars.type())
        interval[1:] = interval_temp
        interval = interval.type(torch.int64)
        
        for batch_idx,idx1,idx2 in zip(np.arange(batch),interval[:-1],interval[1:]):
            this_coords = filtered_coors[idx1:idx2].long()
            indices = this_coords[:,1]*self.xsize +this_coords[:,2]
            batch_pillar =filtered_outs[idx1:idx2].type(pillars.type())
            dynamic_outs = self.dynamic_layer(batch_pillar).softmax(1)
            pseudo_img[batch_idx][indices.long(),:] += batch_pillar
            dynamic_img[batch_idx][indices.long(),:] += dynamic_outs.type(pillars.type())
        
        pseudo_img = pseudo_img.view(batch,self.xsize,self.ysize,n_features).permute(0,3,1,2)
        dynamic_img = dynamic_img.view(batch,self.xsize,self.ysize,3).permute(0,3,1,2) #B,3,H,W
        return pseudo_img,dynamic_img
    
from deformable_conv import DeformableConv2d


class RGB_Net(torch.nn.Module):

    def __init__(self,xsize,ysize,deform=False):
        super().__init__()
        self.xsize = xsize+1
        self.ysize = ysize+1
        self.cnn = resnet_backbone(dims=64)
        # if deform:
        #     self.cnn = deform_resnet_backbone(dims=64)
        # else:
        #     self.cnn = resnet_backbone(dims=64)
            
        self.fpn = BiFPN(64,64)
        
    def forward(self,RGB,dynamic_img,pillar_img_pts,rgb_coors,contains_rgb):
        rgb_out = self.cnn(RGB)
        _,_,c3,c4,c5 = self.fpn(rgb_out)
        batch,C,_,_ = c3.shape
        pseudo_rgb_imga = torch.zeros(batch, C ,self.xsize , self.ysize ).to(RGB.device).type(RGB.type())
        pseudo_rgb_imgb = torch.zeros(batch, C ,self.xsize , self.ysize ).to(RGB.device).type(RGB.type())
        pseudo_rgb_imgc = torch.zeros(batch, C ,self.xsize , self.ysize ).to(RGB.device).type(RGB.type())
        
        # c3.shape,c4.shape,c5.shape(torch.Size([1, 64, 78, 24]),torch.Size([1, 64, 39, 12]),torch.Size([1, 64, 20, 6]))
        # 
        
        for batch_i,(a,b,c,pnts_2d,coors,contain) in enumerate(zip(c3,c4,c5,pillar_img_pts,rgb_coors,contains_rgb)):
            pseudo_rgb_img_filter = torch.ones(1, self.xsize , self.ysize,2).to(RGB.device).type(RGB.type())*-2
            mask = torch.zeros(1, 1, self.xsize , self.ysize).to(RGB.device).type(RGB.type())
            
            contain_filter = contain.to(torch.bool)
            float_indexer = (pnts_2d[contain_filter]*2-1)

            coord = coors[contain_filter].to(torch.int64) # n_pillar,2
            pseudo_rgb_img_filter[0,coord[:,1],coord[:,2],:] = float_indexer
            mask[0,:,coord[:,1],coord[:,2]] = 1

            pseudo_rgb_imga[batch_i] = grid_sample(a.unsqueeze(0),pseudo_rgb_img_filter,align_corners=True)*mask
            pseudo_rgb_imgb[batch_i] = grid_sample(b.unsqueeze(0),pseudo_rgb_img_filter,align_corners=True)*mask
            pseudo_rgb_imgc[batch_i] = grid_sample(c.unsqueeze(0),pseudo_rgb_img_filter,align_corners=True)*mask
            
        pseudo_rgb_img = (pseudo_rgb_imga*dynamic_img[:,0:1]) + (pseudo_rgb_imgb*dynamic_img[:,1:2]) + (pseudo_rgb_imgc*dynamic_img[:,2:3])
        return pseudo_rgb_img
    
# class RGB_Net(torch.nn.Module):
#     '''
#     Neural Network that takes RGB and Dynamic weights as input along with relvant pillar indexes (pillar_img_pts,rgb_coors,contains_rgb)
#     This neural network will return a pseudo img with RGB image features
#     this pseudo img is meant to be concatenated with pointpillar's pseudo image to form a fusion feature (RGB+Pointcloud)
#     '''
#     def __init__(self,xsize,ysize):
#         super().__init__()
#         self.xsize = xsize
#         self.ysize = ysize
#         self.xsize,self.ysize = int(self.xsize)+1,int(self.ysize)+1 
#         self.cnn = resnet_backbone(dims=64)
#         self.fpn = BiFPN(64,64)
        
#     def forward(self,RGB,dynamic_img,pillar_img_pts,rgb_coors,contains_rgb):
#         rgb_out = self.cnn(RGB)
#         _,c3,_,c4,c5 = self.fpn(rgb_out)
#         batch,C,_,_ = c3.shape
#         pseudo_rgb_imga = torch.zeros(batch, C ,self.xsize , self.ysize ).to(RGB.device).type(RGB.type())
#         pseudo_rgb_imgb = torch.zeros(batch, C ,self.xsize , self.ysize ).to(RGB.device).type(RGB.type())
#         pseudo_rgb_imgc = torch.zeros(batch, C ,self.xsize , self.ysize ).to(RGB.device).type(RGB.type())

#         # c3.shape,c4.shape,c5.shape(torch.Size([1, 64, 78, 24]),torch.Size([1, 64, 39, 12]),torch.Size([1, 64, 20, 6]))
#         # 
        
#         for batch_i,(a,b,c,pnts_2d,coors,contain) in enumerate(zip(c3,c4,c5,pillar_img_pts,rgb_coors,contains_rgb)):
#             contain_filter = contain.to(torch.bool)
#             i3 = (pnts_2d[contain_filter]*(torch.tensor([*c3.shape[2:]]).to(RGB.device))).to(torch.int64) #Convert 2D img indexer pnts_2d(0->1) to match the cnn feature
#             i4 = (pnts_2d[contain_filter]*(torch.tensor([*c4.shape[2:]]).to(RGB.device))).to(torch.int64)
#             i5 = (pnts_2d[contain_filter]*(torch.tensor([*c5.shape[2:]]).to(RGB.device))).to(torch.int64)
#             coord = coors[contain_filter].to(torch.int64) # n_pillar,2
#             pseudo_rgb_imga[batch_i,:,coord[:,1],coord[:,2]] += (a[:,i3[:,0],i3[:,1]])
#             pseudo_rgb_imgb[batch_i,:,coord[:,1],coord[:,2]] += (b[:,i4[:,0],i4[:,1]])
#             pseudo_rgb_imgc[batch_i,:,coord[:,1],coord[:,2]] += (c[:,i5[:,0],i5[:,1]])
            
#         pseudo_rgb_img = (pseudo_rgb_imga*dynamic_img[:,0:1]) + (pseudo_rgb_imgb*dynamic_img[:,1:2]) + (pseudo_rgb_imgc*dynamic_img[:,2:3])
#         return pseudo_rgb_img    
    
class NET_4D_EffDet(torch.nn.Module):
    def __init__(self,anchor_dict,n_input_features = 9,n_features = 64,n_pnt_pillar = 100, xyz_range = np.array([0,-40.32,-2,80.64,40.32,3]), 
                  xy_voxel_size= np.array([0.16,0.16]),n_classes=3,rgb_deform=False):
        super().__init__()
        # self.dummy_param = nn.Parameter(torch.empty(0))
        self.voxel_x_grid_size = int((xyz_range[3] - xyz_range[0])//xy_voxel_size[0])
        self.voxel_y_grid_size = int((xyz_range[4] - xyz_range[1])//xy_voxel_size[1])
        
        self.n_features = n_features
        self.SSD = Efficient_Det(anchor_dict,n_pnts_features=128,n_classes=n_classes,xyz_range=xyz_range)
        self.pillar_net = Pillar_Network_SECOND(input_features = n_input_features, n_features=n_features,n_pnts_pillar=n_pnt_pillar)
        self.pillar_to_img = Pseudo_IMG_Scatter_Pillar(self.voxel_x_grid_size,self.voxel_y_grid_size)
        self.rgb_net = RGB_Net(self.voxel_x_grid_size,self.voxel_y_grid_size,deform=rgb_deform)
    
    def forward(self,img,pillar,coord,contains_pillars,pillar_img_pts,rgb_coors,contains_rgb):
        pillar = pillar.permute(0,3,1,2)
        learned_pillars = self.pillar_net(pillar)
        learned_pillars = learned_pillars.squeeze(-1).permute(0,2,1)
        pseudo_img_pillar,dynamic_img = self.pillar_to_img(learned_pillars,coord,contains_pillars)
        pseudo_rgb_img = self.rgb_net(img,dynamic_img,pillar_img_pts,rgb_coors,contains_rgb)
        pseudo_img = torch.cat([pseudo_img_pillar,pseudo_rgb_img],dim=1)
        x = self.SSD(pseudo_img)
        return x,pseudo_img,dynamic_img

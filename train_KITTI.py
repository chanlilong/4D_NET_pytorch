import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from KITTI_dataset import kitti_dataset,KITTI_collate_fn,KITTI_collate_fn_Wcalib
torch.multiprocessing.set_sharing_strategy('file_system')
from pillar_models import NET_4D_EffDet
from tqdm import tqdm
from tensorboardX import SummaryWriter
from itertools import chain
from matcher import Criterion
writer = SummaryWriter('./tensorboard_logs/4dnet_KITTI_SM_maps3')

import matplotlib.pyplot as plt
from  matplotlib.transforms import Affine2D 
from matplotlib.lines import Line2D
import cv2

import pandas as pd

batch_size = 6
xyz_range = np.array([0,-40.32,-2,80.64,40.32,3])
xy_voxel_size= np.array([0.16,0.16])
points_per_pillar = 100
n_pillars=12000

dataset = kitti_dataset(xyz_range = xyz_range,xy_voxel_size= xy_voxel_size,points_per_pillar = points_per_pillar,n_pillars=n_pillars)
dataset_vis = kitti_dataset(xyz_range = xyz_range,xy_voxel_size= xy_voxel_size,points_per_pillar = points_per_pillar,n_pillars=n_pillars,return_calib=True)
data_loader_train = DataLoader(dataset, batch_size=batch_size,collate_fn= KITTI_collate_fn, num_workers=8, shuffle=True)
dataloader_vis = DataLoader(dataset_vis, batch_size=1,collate_fn= KITTI_collate_fn_Wcalib, num_workers=1, shuffle=True)

anchor_dict = np.load("./cluster_kitti_3scales_3anchor.npy",allow_pickle=True).item()
model = NET_4D_EffDet(anchor_dict,n_classes=4)
model.cuda()

model_dict = torch.load("model_KITTI.pth") 
model.load_state_dict(model_dict["params"],strict=False)

criterion = Criterion(4)
model = torch.nn.DataParallel(model, device_ids=[0,1])

n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params:', n_parameters)



optimizer = torch.optim.AdamW(model.parameters(), lr=1e-04,weight_decay=1e-03)
# optimizer.load_state_dict(model_dict["optimizer"])
# optimizer.cuda()
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 35)
# lr_scheduler = model_dict["scheduler"]
# lr_scheduler.cuda()
# scaler = torch.cuda.amp.GradScaler()
# for output bounding box post-processing

def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def draw_rectangle(ax, centre, theta, width, height,color=(1,1,1)):
    
    c, s = np.cos(theta), np.sin(theta)
    R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))

    p1 = [ + width / 2,  + height / 2]
    p2 = [- width / 2,  + height / 2]
    p3 = [ - width / 2, - height / 2]
    p4 = [ + width / 2,  - height / 2]
    p1_new = np.dot(p1, R)+ centre
    p2_new = np.dot(p2, R)+ centre
    p3_new = np.dot(p3, R)+ centre
    p4_new = np.dot(p4, R)+ centre
    
    rect_vertices = np.vstack([p1_new,p2_new,p3_new,p4_new,p1_new]).astype(np.float32)
    line = Line2D(rect_vertices[:,0],rect_vertices[:,1],color=color)
    
    ax.add_line(line)
    
def compute_box_3d(obj, calib):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = roty(obj.yaw)

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.x;
    corners_3d[1, :] = corners_3d[1, :] + obj.y;
    corners_3d[2, :] = corners_3d[2, :] + obj.z;
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = calib.project_rect_to_image(np.transpose(corners_3d))
    # print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)

def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)
    return image    

def return_scaled_boxes(boxes,tgt_boxes,xyz_range):
    boxes[:,0] = boxes[:,0]*((xyz_range[3]-xyz_range[0]))+xyz_range[0]
    boxes[:,1] = boxes[:,1]*((xyz_range[4]-xyz_range[1]))+xyz_range[1]
    boxes[:,2] = boxes[:,2]*((xyz_range[5]-xyz_range[2]))+xyz_range[2]
    boxes[:,3] = boxes[:,3]*((xyz_range[3]-xyz_range[0]))
    boxes[:,4] = boxes[:,4]*((xyz_range[4]-xyz_range[1]))
    boxes[:,5] = boxes[:,5]*((xyz_range[5]-xyz_range[2]))
    boxes[:,6] = (boxes[:,6]*2 - 1)*np.pi

    tgt_boxes[:,0] = tgt_boxes[:,0]*((xyz_range[3]-xyz_range[0]))+xyz_range[0]
    tgt_boxes[:,1] = tgt_boxes[:,1]*((xyz_range[4]-xyz_range[1]))+xyz_range[1]
    tgt_boxes[:,2] = tgt_boxes[:,2]*((xyz_range[5]-xyz_range[2]))+xyz_range[2]
    tgt_boxes[:,3] = tgt_boxes[:,3]*((xyz_range[3]-xyz_range[0]))
    tgt_boxes[:,4] = tgt_boxes[:,4]*((xyz_range[4]-xyz_range[1]))
    tgt_boxes[:,5] = tgt_boxes[:,5]*((xyz_range[5]-xyz_range[2]))
    tgt_boxes[:,6] = (tgt_boxes[:,6]*2 - 1)*np.pi
    
    return boxes,tgt_boxes


def write_to_tensorboard(itr,metrics,writer,log_type="train_losses/",detach=True):
    for key,value in metrics.items():
        name = log_type + key
        if detach:
            writer.add_scalar(name, value.detach().cpu().mean().numpy(), itr)
        else:
            writer.add_scalar(name, value.cpu().numpy(), itr)

def detect(img, pillars, coord, contains_pillars,pillar_img_pts,rgb_coors,contains_rgb, model):


    return probas[keep].detach().cpu().numpy(), pred_boxes.detach().cpu().numpy(), outputs, pseudo_img,dynamic_img,img
def show_model_inference():

    model.eval()
    with torch.no_grad():
        for img,(pillars, coord, contains_pillars),(pillar_img_pts,rgb_coors,contains_rgb),targets,calibs in dataloader_vis:
            break
        outputs,pseudo_img,dynamic_img = model(img.cuda(),pillars.float().cuda(), coord.cuda(), contains_pillars.cuda(),pillar_img_pts.float().cuda(),rgb_coors.cuda(),contains_rgb.cuda())

    probas,_ = outputs['pred_logits'][0, :, 0:].sigmoid().max(-1)
    keep = (probas >= 0.2).squeeze()

    pred_boxes = outputs['pred_boxes'][0, keep]

    pred_boxes,target_boxes = return_scaled_boxes(pred_boxes.cpu().numpy(),targets[0]["boxes"].numpy(),xyz_range)

    target_boxes_df = pd.DataFrame(target_boxes,columns=["z","x","y","w","l","h","yaw"])
    
    pred_boxes_df = pd.DataFrame(pred_boxes,columns=["z","x","y","w","l","h","yaw"])
    

    target_boxes_df["x"] *=-1
    pred_boxes_df["x"] *=-1

    tgt_boxes = []
    pred_boxes = []
    for _,row in target_boxes_df.iterrows():
        tgt_box,_ = compute_box_3d(row,calibs[0])
        if type(tgt_box) != type(None):
            tgt_boxes.append(tgt_box)

    for _,row in pred_boxes_df.iterrows():
        pred_box,_ = compute_box_3d(row,calibs[0])
        if type(pred_box) != type(None):
            pred_boxes.append(pred_box)
                
    img2 = img[0].permute(1,2,0).numpy().copy()
    img2 = cv2.resize(img2,(1242,375))
    
    for b in tgt_boxes:  
        img2 = draw_projected_box3d(img2,b,color=(255,0,0))

    for b in pred_boxes:  
        img2 = draw_projected_box3d(img2,b,color=(0,255,0))
        
    fig,ax = plt.subplots(1,1,figsize=(15,10))
    ax.imshow(img2)
    
    pillars2 = pillars[0,contains_pillars.type(torch.bool).flatten()]
    x = (pillars2[...,0]*(xyz_range[3]-xyz_range[0]))+xyz_range[0]
    y = (pillars2[...,1]*(xyz_range[4]-xyz_range[1]))+xyz_range[1]
    z = (pillars2[...,2]*(xyz_range[5]-xyz_range[2]))+xyz_range[2]
    # i = pillars2[...,3]
    # r = np.sqrt(x**2+y**2)
    # r = ((r-r.min())/(r.max()-r.min())) + ((i-i.min())/(i.max()-i.min())) 

    fig2,ax2 = plt.subplots(1,1,figsize=(10,10))

    ax2.scatter(x,y,s=1,c=z,cmap="jet")
    ax2.set_facecolor((0,0,0))
    
    target_boxes_df["x"] *=-1
    pred_boxes_df["x"] *=-1
    
    for b in target_boxes_df.values:
        x,y,z,w,l,h,r = b
        # y *= -1
        draw_rectangle(ax2, (x,y), r, w, l,color=(1,0,0))
        
    for b in pred_boxes_df.values:
        x,y,z,w,l,h,r = b
        # y *= -1
        draw_rectangle(ax2, (x,y), r, w, l,color=(0,1,0))
        
    ax2.axis('tight')
    ax2.set_facecolor((0,0,0))
    ax2.set_xlim([0,80])
    ax2.set_ylim([-40.32,40.32])

    fig3,ax3 = plt.subplots(figsize=(10,10))
    ax3.imshow(pseudo_img[0,0:64].mean(0).detach().cpu().numpy(),cmap="jet")
    ax3.axis("tight")
    ax3.invert_yaxis()
    
    fig4,ax4 = plt.subplots(figsize=(10,10))
    ax4.imshow(pseudo_img[0,64:].mean(0).detach().cpu().numpy(),cmap="jet")
    ax4.axis("tight")
    ax4.invert_yaxis()
    
    fig5,ax5 = plt.subplots(figsize=(10,10))
    ax5.imshow(dynamic_img[0,0].detach().cpu().numpy(),cmap="jet") #max values across channels
    ax5.axis("tight")
    ax5.invert_yaxis()
    
    model.train()
    return fig,fig2,fig3,fig4,fig5

itr=0
itr = model_dict["itr"]
# max_norm = args.clip_max_norm

model.train()
model.cuda()
for e in tqdm(range(50)):

    for i,(img,(pillars, coord, contains_pillars),(pillar_img_pts,rgb_coors,contains_rgb),targets) in enumerate(data_loader_train):
        
        # with torch.cuda.amp.autocast():
        pred,_,_= model(img.cuda(),pillars.float().cuda(), coord.cuda(), contains_pillars.cuda(),pillar_img_pts.float().cuda(),rgb_coors.cuda(),contains_rgb.cuda())

        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        loss_dict = criterion(pred, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        max_probs = pred['pred_logits'][:,:, 0:].sigmoid().max()
            
            
        
        losses.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # scaler.scale(losses).backward()
        # scaler.step(optimizer)
        # scaler.update()
        # optimizer.zero_grad()

        itr+=1
        
        if itr%2==0:
            with torch.no_grad():
                write_to_tensorboard(itr,loss_dict,writer)
                writer.add_scalar("train_losses/max_probability", max_probs.detach().cpu().numpy(), itr)
                #freeze_detr(model,freeze=False)
        if itr%50==0:
            fig,fig2,fig3,fig4,fig5 = show_model_inference()
            writer.add_figure("images/front_view",fig,itr)
            writer.add_figure("images/predicted",fig2,itr)
            writer.add_figure("images/LiDAR_Pseudoimg",fig3,itr)
            writer.add_figure("images/RGB_Pseudoimg",fig4,itr)
            writer.add_figure("images/dynamic_img",fig5,itr)

        if itr%50==0 and itr!=0:
            torch.cuda.empty_cache()     
            
        if itr%500==0 and e!=0:
            model_dict = {"params":model.module.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":lr_scheduler,"itr":itr}
            torch.save(model_dict,"./model_KITTI.pth")
            # break
    lr_scheduler.step()
    
model_dict = {"params":model.module.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":lr_scheduler,"itr":itr}
torch.save(model_dict,"./model_KITTI.pth")
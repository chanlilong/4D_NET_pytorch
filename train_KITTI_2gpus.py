import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from KITTI_dataset import kitti_dataset,KITTI_collate_fn
torch.multiprocessing.set_sharing_strategy('file_system')
from pillar_models import NET_4D_EffDet
from tqdm import tqdm
from tensorboardX import SummaryWriter
from itertools import chain
from matcher import Criterion
writer = SummaryWriter('./tensorboard_logs/4dnet_KITTI_SM_maps2')

import matplotlib.pyplot as plt
from  matplotlib.transforms import Affine2D 
from matplotlib.lines import Line2D


batch_size = 6
xyz_range = np.array([0,-40.32,-2,80.64,40.32,3])
xy_voxel_size= np.array([0.16,0.16])
points_per_pillar = 100
n_pillars=12000

dataset = kitti_dataset(xyz_range = xyz_range,xy_voxel_size= xy_voxel_size,points_per_pillar = points_per_pillar,n_pillars=n_pillars)
data_loader_train = DataLoader(dataset, batch_size=batch_size,collate_fn= KITTI_collate_fn, num_workers=8, shuffle=True)
dataloader_vis = DataLoader(dataset, batch_size=1,collate_fn= KITTI_collate_fn, num_workers=1, shuffle=True)

anchor_dict = np.load("./cluster_kitti_3scales_3anchor.npy",allow_pickle=True).item()
model = NET_4D_EffDet(anchor_dict,n_classes=4)
model.cuda()

# model_dict = torch.load("model_KITTI.pth") 
# model.load_state_dict(model_dict["params"],strict=False)

criterion = Criterion(4)
model = torch.nn.DataParallel(model, device_ids=[0,1])

n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params:', n_parameters)



optimizer = torch.optim.AdamW(model.parameters(), lr=1e-04,weight_decay=1e-03)
# optimizer.load_state_dict(model_dict["optimizer"])
# optimizer.cuda()
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25)
# lr_scheduler = model_dict["scheduler"]
# lr_scheduler.cuda()
# scaler = torch.cuda.amp.GradScaler()
# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes_kitti(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).cuda()
    return b



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
    
def plot_results(pillars,contains_pillars, prob, boxes,tgt_boxes,pseudo_img,dynamic_img):
    

    pillars = pillars[contains_pillars.astype(bool).flatten()]
    x = (pillars[...,0]*(xyz_range[3]-xyz_range[0]))+xyz_range[0]
    y = (pillars[...,1]*(xyz_range[4]-xyz_range[1]))+xyz_range[1]
    z = (pillars[...,2]*(xyz_range[5]-xyz_range[2]))+xyz_range[2]
    i = pillars[...,3]
    r = np.sqrt(x**2+y**2)
    r = ((r-r.min())/(r.max()-r.min())) + ((i-i.min())/(i.max()-i.min())) 

    fig,ax = plt.subplots(1,1,figsize=(10,10))

    ax.scatter(x,y,s=1,c=r,cmap="jet")


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

    for b in boxes:

        xc,yc,zc,l,w,h,rot = b
        draw_rectangle(ax,[xc,yc],rot,l,w,color=(0,1,0))

    
    for b in tgt_boxes:

        xc,yc,zc,l,w,h,rot = b
        draw_rectangle(ax,[xc,yc],rot,l,w,color=(1,0,0))



    ax.axis('tight')
    ax.set_facecolor((0,0,0))
    ax.set_xlim([0,80])
    ax.set_ylim([-40.32,40.32])

    fig2,ax2 = plt.subplots(figsize=(10,10))
    ax2.imshow(pseudo_img[0].mean(0).detach().cpu().numpy(),cmap="jet")
    ax2.axis("tight")
    ax2.invert_yaxis()
    
    fig3,ax3 = plt.subplots(figsize=(10,10))
    ax3.imshow(pseudo_img[0,64:].mean(0).detach().cpu().numpy(),cmap="jet")
    ax3.axis("tight")
    ax3.invert_yaxis()
    
    fig4,ax4 = plt.subplots(figsize=(10,10))
    ax4.imshow(dynamic_img[0].max(0).values.detach().cpu().numpy(),cmap="jet") #max values across channels
    ax4.axis("tight")
    ax4.invert_yaxis()

    
    return fig,fig2,fig3,fig4

def write_to_tensorboard(itr,metrics,writer,log_type="train_losses/",detach=True):
    for key,value in metrics.items():
        name = log_type + key
        if detach:
            writer.add_scalar(name, value.detach().cpu().mean().numpy(), itr)
        else:
            writer.add_scalar(name, value.cpu().numpy(), itr)

def detect(img, pillars, coord, contains_pillars,pillar_img_pts,rgb_coors,contains_rgb, model):
    # propagate through the model
    # outputs,pseudo_img = model(pillars.float().cuda(),coords.cuda(),contains_pillars.cuda())
    outputs,pseudo_img,dynamic_img = model(img.cuda(),pillars.float().cuda(), coord.cuda(), contains_pillars.cuda(),pillar_img_pts.float().cuda(),rgb_coors.cuda(),contains_rgb.cuda())
    # keep only predictions with 0.7+ confidence
    probas,_ = outputs['pred_logits'][0, :, 0:].sigmoid().max(-1)
    keep = (probas >= 0.2).squeeze()
    # convert boxes from [0; 1] to image scales
    pred_boxes = outputs['pred_boxes'][0, keep]

    return probas[keep].detach().cpu().numpy(), pred_boxes.detach().cpu().numpy(), outputs, pseudo_img,dynamic_img
def show_model_inference():

    model.eval()
    with torch.no_grad():
        for img,(pillars, coord, contains_pillars),(pillar_img_pts,rgb_coors,contains_rgb),outputs in dataloader_vis:
            break
        # rand_idx = np.random.randint(0,len(dataset))
        # ([pillars,coords,contains_pillars],targets) = dataset[rand_idx]
        scores, boxes,_ ,pseudo_img,dynamic_img= detect(img, pillars, coord, contains_pillars,pillar_img_pts,rgb_coors,contains_rgb, model)
        fig,fig2,fig3,fig4 = plot_results(pillars[0].cpu().numpy(),contains_pillars[0].cpu().numpy(), scores, boxes,outputs[0]["boxes"], pseudo_img,dynamic_img)
    model.train()
    return fig,fig2,fig3,fig4

itr=0
# itr = model_dict["itr"]
# max_norm = args.clip_max_norm

model.train()
model.cuda()
for e in tqdm(range(30)):

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
        
        if itr%5==0:
            with torch.no_grad():
                write_to_tensorboard(itr,loss_dict,writer)
                writer.add_scalar("train_losses/max_probability", max_probs.detach().cpu().numpy(), itr)
                #freeze_detr(model,freeze=False)
        if itr%50==0:
            fig,fig2,fig3,fig4 = show_model_inference()
            writer.add_figure("images/predicted",fig,itr)
            writer.add_figure("images/pseudo_img_mean",fig2,itr)
            writer.add_figure("images/rgb_feature_img",fig3,itr)
            writer.add_figure("images/dynamic_weight_img",fig4,itr)

        if itr%50==0 and itr!=0:
            torch.cuda.empty_cache()     
            
        if itr%500==0 and e!=0:
            model_dict = {"params":model.module.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":lr_scheduler,"itr":itr}
            torch.save(model_dict,"./model_KITTI.pth")
            # break
    lr_scheduler.step()
    
model_dict = {"params":model.module.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":lr_scheduler,"itr":itr}
torch.save(model_dict,"./model_KITTI.pth")
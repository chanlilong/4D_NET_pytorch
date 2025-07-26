from __future__ import annotations

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.KITTI_dataset import KITTI_collate_fn
from dataset.KITTI_dataset import KITTI_collate_fn_Wcalib
from dataset.KITTI_dataset import KittiDataset
from models.assigner.matcher import Criterion
from models.pillar_models import NET_4D_EffDet
from utils.draw_utils import compute_box_3d
from utils.draw_utils import draw_projected_box3d
from utils.draw_utils import draw_rectangle

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

# torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
writer = SummaryWriter('./tensorboard_logs/4dnet_KITTI')
batch_size = 20
xyz_range = np.array([0, -40.32, -2, 80.64, 40.32, 3])
xy_voxel_size = np.array([0.16, 0.16])
points_per_pillar = 32
n_pillars = 12000

dataset = KittiDataset(
    xyz_range=xyz_range,
    xy_voxel_size=xy_voxel_size,
    points_per_pillar=points_per_pillar,
    n_pillars=n_pillars,
)
dataset_vis = KittiDataset(
    xyz_range=xyz_range,
    xy_voxel_size=xy_voxel_size,
    points_per_pillar=points_per_pillar,
    n_pillars=n_pillars,
    return_calib=True,
)
data_loader_train = DataLoader(
    dataset,
    batch_size=batch_size,
    collate_fn=KITTI_collate_fn,
    num_workers=8,
    shuffle=True,
)
dataloader_vis = DataLoader(
    dataset_vis,
    batch_size=1,
    collate_fn=KITTI_collate_fn_Wcalib,
    num_workers=1,
    shuffle=True,
)

anchor_dict = np.load('./cluster_kitti_3scales_3anchor.npy', allow_pickle=True).item()
model = NET_4D_EffDet(
    anchor_dict,
    n_classes=4,
    xyz_range=xyz_range,
    n_pnt_pillar=points_per_pillar,
    xy_voxel_size=xy_voxel_size,
    rgb_deform=False,
)


model_dict = torch.load('./weights/model_KITTI_exp_CP2.pth')
model.load_state_dict(model_dict['params'], strict=False)

criterion = Criterion(4)
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
model.cuda()
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params:', n_parameters)


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-04)
optimizer.load_state_dict(model_dict['optimizer'])
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15)
lr_scheduler.load_state_dict(model_dict['scheduler'])


def write_to_tensorboard(itr, metrics, writer, log_type='train_losses/', detach=True):
    for key, value in metrics.items():
        name = log_type + key
        if detach:
            writer.add_scalar(name, value.detach().cpu().mean().numpy(), itr)
        else:
            writer.add_scalar(name, value.cpu().numpy(), itr)


def show_model_inference():
    model.eval()
    with torch.no_grad():
        for img, (pillars, coord, contains_pillars), (
            pillar_img_pts,
            rgb_coors,
            contains_rgb,
        ), targets, calibs in dataloader_vis:
            break
        outputs, pseudo_img, dynamic_img = model(
            img.cuda(),
            pillars.float().cuda(),
            coord.cuda(),
            contains_pillars.cuda(),
            pillar_img_pts.float().cuda(),
            rgb_coors.cuda(),
            contains_rgb.cuda(),
        )

    probas, _ = outputs['pred_logits'][0, :, 0:].sigmoid().max(-1)
    keep = (probas >= 0.2).squeeze()

    pred_boxes = outputs['pred_boxes'][0, keep]

    pred_boxes, target_boxes = (pred_boxes.cpu().numpy(), targets[0]['boxes'].numpy())

    target_boxes_df = pd.DataFrame(
        target_boxes,
        columns=['z', 'x', 'y', 'l', 'w', 'h', 'yaw'],
    )

    pred_boxes_df = pd.DataFrame(
        pred_boxes,
        columns=['z', 'x', 'y', 'l', 'w', 'h', 'yaw'],
    )

    target_boxes_df['x'] *= -1
    pred_boxes_df['x'] *= -1

    tgt_boxes = []
    pred_boxes = []
    for _, row in target_boxes_df.iterrows():
        tgt_box, _ = compute_box_3d(row, calibs[0])
        if tgt_box is not None:
            tgt_boxes.append(tgt_box)

    for _, row in pred_boxes_df.iterrows():
        pred_box, _ = compute_box_3d(row, calibs[0])
        if pred_box is not None:
            pred_boxes.append(pred_box)

    img2 = img[0].permute(1, 2, 0).numpy().copy()
    img2 = cv2.resize(img2, (1242, 375))
    img2 *= 255.0
    img2 = img2.astype(np.uint8)

    for b in tgt_boxes:
        img2 = draw_projected_box3d(img2, b, color=(255, 0, 0))

    for b in pred_boxes:
        img2 = draw_projected_box3d(img2, b, color=(0, 255, 0))

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(img2.astype(np.uint8))

    pillars2 = pillars[0, contains_pillars.type(torch.bool).flatten()]
    x = (pillars2[..., 0] * (xyz_range[3] - xyz_range[0])) + xyz_range[0]
    y = (pillars2[..., 1] * (xyz_range[4] - xyz_range[1])) + xyz_range[1]
    z = (pillars2[..., 2] * (xyz_range[5] - xyz_range[2])) + xyz_range[2]

    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))

    ax2.scatter(x, y, s=1, c=z, cmap='jet')
    ax2.set_facecolor((0, 0, 0))

    target_boxes_df['x'] *= -1
    pred_boxes_df['x'] *= -1

    for b in target_boxes_df.to_numpy():
        x, y, z, l, w, h, r = b
        draw_rectangle(ax2, (x, y), r, w, l, color=(1, 0, 0))

    for b in pred_boxes_df.to_numpy():
        x, y, z, l, w, h, r = b
        draw_rectangle(ax2, (x, y), r, w, l, color=(0, 1, 0))

    ax2.axis('tight')
    ax2.set_facecolor((0, 0, 0))
    ax2.set_xlim([0, 80])
    ax2.set_ylim([-40.32, 40.32])

    fig3, ax3 = plt.subplots(figsize=(10, 10))
    ax3.imshow(
        pseudo_img[0, 0:64].mean(0).detach().cpu().numpy(),
        cmap='jet',
    )  # From Pillars
    ax3.axis('tight')
    ax3.invert_yaxis()

    fig4, ax4 = plt.subplots(figsize=(10, 10))
    ax4.imshow(
        pseudo_img[0, 64:].max(0).values.detach().cpu().numpy(),
        cmap='jet',
    )  # From RGB
    ax4.axis('tight')
    ax4.invert_yaxis()

    fig5, ax5 = plt.subplots(figsize=(10, 10))
    ax5.imshow(
        dynamic_img[0, 0].detach().cpu().numpy(),
        cmap='jet',
    )  # max values across channels
    ax5.axis('tight')
    ax5.invert_yaxis()

    model.train()
    return fig, fig2, fig3, fig4, fig5


itr = 0
itr = model_dict['itr']

for e in tqdm(range(25)):
    for i, (
        img,
        (pillars, coord, contains_pillars),
        (pillar_img_pts, rgb_coors, contains_rgb),
        targets,
    ) in enumerate(data_loader_train):
        for param in model.parameters():
            param.grad = None

        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        # with torch.cuda.amp.autocast():
        pred, _, _ = model(
            img.cuda(),
            pillars.float().cuda(),
            coord.cuda(),
            contains_pillars.cuda(),
            pillar_img_pts.float().cuda(),
            rgb_coors.cuda(),
            contains_rgb.cuda(),
        )

        loss_dict = criterion(pred, targets)
        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        max_probs = pred['pred_logits'][:, :, 0:].sigmoid().max()

        losses.backward()
        optimizer.step()
        optimizer.zero_grad()

        if itr % 2 == 0:
            with torch.no_grad():
                write_to_tensorboard(itr, loss_dict, writer)
                writer.add_scalar(
                    'train_losses/max_probability',
                    max_probs.detach().cpu().numpy(),
                    itr,
                )
                # freeze_detr(model,freeze=False)
        if itr % 100 == 0:
            fig, fig2, fig3, fig4, fig5 = show_model_inference()
            writer.add_figure('images/front_view', fig, itr)
            writer.add_figure('images/predicted', fig2, itr)
            writer.add_figure('images/LiDAR_Pseudoimg', fig3, itr)
            writer.add_figure('images/RGB_Pseudoimg', fig4, itr)
            writer.add_figure('images/dynamic_img', fig5, itr)

        if itr % 250 == 0 and itr != 0:
            torch.cuda.empty_cache()

        if itr % 250 == 0 and itr != 0:
            model_dict = {
                'params': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'itr': itr,
                'xyz_range': xyz_range,
                'xy_voxel_size': xy_voxel_size,
                'pnts_per_pillar': points_per_pillar,
                'n_pillars': n_pillars,
            }
            torch.save(model_dict, './weights/model_KITTI_exp.pth')
            # break
        itr += 1


model_dict = {
    'params': model.module.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': lr_scheduler.state_dict(),
    'itr': itr,
    'xyz_range': xyz_range,
    'xy_voxel_size': xy_voxel_size,
    'pnts_per_pillar': points_per_pillar,
    'n_pillars': n_pillars,
}
torch.save(model_dict, './weights/model_KITTI_exp.pth')

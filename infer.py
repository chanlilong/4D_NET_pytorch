from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mmcv.ops.iou3d import nms3d_normal
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.KITTI_dataset import KITTI_collate_fn_Wcalib
from dataset.KITTI_dataset import KittiDataset
from models.pillar_models import NET_4D_EffDet

if __name__ == '__main__':
    xyz_range = np.array([0, -40.32, -2, 80.64, 40.32, 3])
    xy_voxel_size = np.array([0.16, 0.16])
    points_per_pillar = 32
    n_pillars = 12000

    anchor_dict = np.load(
        './anchors/cluster_kitti_3scales_3anchor.npy',
        allow_pickle=True,
    ).item()
    model_dict = torch.load('./weights/model_KITTI_exp.pth', weights_only=False)
    model = NET_4D_EffDet(
        anchor_dict,
        n_classes=4,
        n_pnt_pillar=points_per_pillar,
        xyz_range=xyz_range,
        xy_voxel_size=xy_voxel_size,
    )
    model.load_state_dict(model_dict['params'], strict=True)
    model.cuda()
    model.eval()

    dataset = KittiDataset(
        root='/mnt/4TB/Datasets/kitti_dataset/training/',
        xyz_range=xyz_range,
        xy_voxel_size=xy_voxel_size,
        points_per_pillar=points_per_pillar,
        n_pillars=n_pillars,
        return_calib=True,
        test=True,
    )
    dataloader_vis = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=KITTI_collate_fn_Wcalib,
        num_workers=0,
        shuffle=True,
    )

    pred_dfs = pd.DataFrame(
        columns=['class', 'x', 'y', 'z', 'l', 'w', 'h', 'r', 'score'],
    )
    gt_dfs = pd.DataFrame(columns=['class', 'x', 'y', 'z', 'l', 'w', 'h', 'r', 'score'])

    int2clss = {0: 'car', 1: 'pedestrian', 2: 'cyclist', 3: 'misc'}
    for idx, (
        img,
        (pillars, coord, contains_pillars),
        (pillar_img_pts, rgb_coors, contains_rgb),
        targets,
        _,
    ) in tqdm(enumerate(dataloader_vis), total=len(dataloader_vis)):
        target = targets[0]
        gt_boxes = target['boxes'].view(-1, 7)
        gt_labels = target['labels'].view(-1, 1)
        gt_labels = [int2clss[x] for x in gt_labels.numpy().flatten()]
        gt_df = pd.DataFrame(
            gt_boxes.numpy(),
            columns=['x', 'y', 'z', 'l', 'w', 'h', 'r'],
        )

        gt_df['class'] = gt_labels
        gt_df['score'] = np.ones(gt_df.shape[0])

        gt_df = gt_df[['class', 'x', 'y', 'z', 'l', 'w', 'h', 'r', 'score']]
        gt_df.to_csv(
            f'./prediction_eval/gt/{idx:06d}.csv',
            sep=',',
            index=False,
            header=False,
        )

        with torch.no_grad():
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
        keep = (probas >= 0.1).squeeze()
        pred_boxes = outputs['pred_boxes'][0, keep].detach().cpu()
        pred_clss = (
            outputs['pred_logits'][0, :, 0:].sigmoid()[keep].argmax(-1).cpu().numpy()
        )
        pred_score = probas[keep.type(torch.bool)].float()

        if len(pred_boxes) > 0:
            pred_idx = (
                nms3d_normal(pred_boxes.cuda(), pred_score, 0.1)
                .cpu()
                .numpy()
                .astype(np.int64)
            )
            pred_boxes = pred_boxes[pred_idx]
            pred_score = pred_score[pred_idx]
            pred_clss = [int2clss[x] for x in pred_clss[pred_idx]]

        pred_df = pd.DataFrame(
            pred_boxes.numpy(),
            columns=['x', 'y', 'z', 'l', 'w', 'h', 'r'],
        )

        pred_df['class'] = pred_clss
        pred_df['score'] = pred_score.cpu().numpy()

        pred_df = pred_df[['class', 'x', 'y', 'z', 'l', 'w', 'h', 'r', 'score']]

        save_path = Path(f'./prediction_eval/pred/{idx:06d}.csv')
        if not save_path.parent.exists():
            save_path.parent.mkdir()
        pred_df.to_csv(
            save_path,
            sep=',',
            index=False,
            header=False,
        )

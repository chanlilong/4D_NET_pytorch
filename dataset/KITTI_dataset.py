from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from point_cloud_ops import points_to_voxel
from reader import Label3D

pd.options.mode.chained_assignment = None

import cv2
from PIL import Image

from calibration_util import Calibration


class kitti_dataset(Dataset):
    '''
    Kitti Dataset meant for 4D Net: https://ai.googleblog.com/2022/02/4d-net-learning-multi-modal-alignment.html
    
    Returns :img,(pillars, coord, contains_pillars),(pillar_img_pts2,pillar_img_filter),outputs
    Explaination:
            img:     Image_2 rgb image from kitti
            pillars: [12000,100,9],pillars are voxelized pointclouds,
            coord:   [12000,2], indexes that locate where pillars fall onto the pseudo_image
            contains_pillars: [12000], boolean array (0,1) that identifies active pillars
            
            pillar_img_pts2: [12000,2], normalized pixel coordinates (0,1) that is meant for indexing RGB feature maps
            pillar_img_filer: [12000], boolean array (0,1) that identifies active pillars, this will mean not all active pillars (contains_pillars) will contain rgb_features
            
            outputs: sample targets ["boxes" (x,y,z,w,l,h,r),"clss (ped,cyc,car)"]
    '''

    def __init__(self, root='/home/conda/RAID_5_14TB/DATASETS/KITTI_dataset/training/',
                xyz_range=np.array([0, -40.32, -2, 80.64, 40.32, 3]),
                xy_voxel_size=np.array([0.16, 0.16]), points_per_pillar=100, n_pillars=12000, return_calib=False, stx=False, test=False):
        super().__init__()

        test_set_names = np.load('testset_names.npy')
        pc_root = root + 'velodyne/'
        if stx:
            pc_root = root + 'stx/'
        images_root = root + 'image_2/'
        labels_root = root + 'label_2/'
        calib_root = root + 'calib/'
        pc_filenames = glob.glob(pc_root + '*.bin')
        frame_names = [x.split('/')[-1].split('.')[0] for x in pc_filenames]
        self.return_calib = return_calib

        self.images_train = [images_root + f'{x}.png' for x in frame_names if x not in test_set_names]
        self.pc_filenames_train = [pc_root + f'{x}.bin' for x in frame_names if x not in test_set_names]
        self.calib_train = [calib_root + f'{x}.txt' for x in frame_names if x not in test_set_names]
        self.labels_train = [labels_root + f'{x}.txt' for x in frame_names if x not in test_set_names]
        self.pc_filenames_test = [pc_root + f'{x}.bin' for x in frame_names if x in test_set_names]
        self.labels_test = [labels_root + f'{x}.txt' for x in frame_names if x in test_set_names]
        self.train = True
        if test:
            self.pc_filenames_train = self.pc_filenames_test
            self.labels_train = self.labels_test
            self.images_train = [images_root + f'{x}.png' for x in frame_names if x in test_set_names]
            self.train = False

        self.n_pillars = n_pillars
        self.points_per_pillar = points_per_pillar
        self.xyz_range = xyz_range
        self.xy_voxel_size = xy_voxel_size
        self.x_size = (self.xyz_range[3] - self.xyz_range[0]) // self.xy_voxel_size[0]
        self.y_size = (self.xyz_range[4] - self.xyz_range[1]) // self.xy_voxel_size[1]
        self.x_offset = self.xy_voxel_size[0] / 2 + self.xyz_range[0]
        self.y_offset = self.xy_voxel_size[1] / 2 + self.xyz_range[1]

        self.classes = {'Car':               0,
                           'Pedestrian':        1,
                           'Person_sitting':    1,
                           'Cyclist':           2,
                           'Truck':             0,
                           'Van':               0,
                           'Tram':              3,
                           'Misc':              3,
                        }

        self.classes_anchor = {
                                'Car': np.array([1.6, 3.9, 1.56]),
                                'Pedestrian': np.array([0.6, 0.8, 1.73]),
                                'Cyclist': np.array([0.6, 1.76, 1.73]),
                                }

        self.col_names = ['type', 'truncated', 'occluded', 'alpha', 'x1', 'y1', 'x2', 'y2', 'h', 'w', 'l', 'x', 'y', 'z', 'yaw']

    @staticmethod
    def transform_labels_into_lidar_coordinates(labels: list[Label3D], R: np.ndarray, t: np.ndarray):
        """
        Input: label3D, Rotation Matrix, Translation Vector

        Output: lidar coordinates
        """
        transformed = []
        for label in labels:
            label.centroid = label.centroid @ np.linalg.inv(R).T - t
            label.dimension = label.dimension[[2, 1, 0]]
            label.yaw -= np.pi / 2
            while label.yaw < -np.pi:
                label.yaw += (np.pi * 2)
            while label.yaw > np.pi:
                label.yaw -= (np.pi * 2)
            transformed.append(label)
        return transformed

    def get_calib(self, idx):

        calib_df = pd.read_csv(self.calib_train[idx], sep=' ', header=None)
        calib_df.index = [x.replace(':', '') for x in calib_df.iloc[:, 0].values]
        calib_df = calib_df.drop(0, axis=1)
        Tr_velo_to_cam = calib_df.loc['Tr_velo_to_cam'].values.reshape(3, 4)

        R, t = Tr_velo_to_cam[:, :3], Tr_velo_to_cam[:, 3]

        return R, t

    def __getitem__(self, idx):

        if self.return_calib:
            success, img, (pillars, coord, contains_pillars), (pillar_img_pts2, rgb_coors, contains_rgb), outputs, calib = self.get_func(idx)

        else:
            success, img, (pillars, coord, contains_pillars), (pillar_img_pts2, rgb_coors, contains_rgb), outputs = self.get_func(idx)

        if not success:
            while not success:
                idx = np.random.randint(0, len(self))
                if self.return_calib:
                    success, img, (pillars, coord, contains_pillars), (pillar_img_pts2, rgb_coors, contains_rgb), outputs, calib = self.get_func(idx)
                else:
                    success, img, (pillars, coord, contains_pillars), (pillar_img_pts2, rgb_coors, contains_rgb), outputs = self.get_func(idx)

        if self.return_calib:
            return img, (pillars, coord, contains_pillars), (pillar_img_pts2, rgb_coors, contains_rgb), outputs, calib
        else:
            return img, (pillars, coord, contains_pillars), (pillar_img_pts2, rgb_coors, contains_rgb), outputs

    def get_func(self, idx):
        try:
            if self.return_calib:
                img, (pillars, coord, contains_pillars), (pillar_img_pts2, rgb_coors, contains_rgb), outputs, calib = self.getitem(idx)
                return True, img, (pillars, coord, contains_pillars), (pillar_img_pts2, rgb_coors, contains_rgb), outputs, calib
            else:
                img, (pillars, coord, contains_pillars), (pillar_img_pts2, rgb_coors, contains_rgb), outputs = self.getitem(idx)
                return True, img, (pillars, coord, contains_pillars), (pillar_img_pts2, rgb_coors, contains_rgb), outputs

            if self.train:
                assert len(outputs['boxes']) > 0, 'num samples must be more than 0 (training purposes)'
        except:
            # print("hi")
            if self.return_calib:
                return False, None, (None, None, None), (None, None, None), None, None
            else:
                return False, None, (None, None, None), (None, None, None), None

    def getitem(self, idx):
        img_file = self.images_train[idx]
        pc_file = self.pc_filenames_train[idx]
        label_file = self.labels_train[idx]
        calib_file = self.calib_train[idx]
        calib = Calibration(calib_file)
        # get input
        # pointcloud = KittiDataReader.read_lidar(pc_file)
        img = Image.open(img_file).convert('RGB')
        img = img.resize((1242, 375))
        img = np.asarray(img).astype(np.uint8)
        H, W, C = img.shape
        pointcloud = np.fromfile(pc_file, dtype=np.float32).reshape((-1, 4))
        pc_filter = (pointcloud[:, 0] > self.xyz_range[0]) & (pointcloud[:, 0] < self.xyz_range[3]) \
        & (pointcloud[:, 1] > self.xyz_range[1]) & (pointcloud[:, 1] < self.xyz_range[4]) \
        & (pointcloud[:, 2] > self.xyz_range[2]) & (pointcloud[:, 2] < self.xyz_range[5])
        pointcloud = pointcloud[pc_filter]
        np.random.shuffle(pointcloud)
        # 1.Voxelize pc to pillars
        voxels, coors, num_points_per_voxel = points_to_voxel(pointcloud, voxel_size=[*self.xy_voxel_size, 4], coors_range=[*self.xyz_range], max_points=self.points_per_pillar, max_voxels=self.n_pillars)

        # 2.Init pillar coordinates and pillar boolean indicator (contains_pillar)
        coord = np.zeros((self.n_pillars, 3))
        coord[:voxels.shape[0], :] = coors

        contains_pillars = np.zeros(self.n_pillars)
        contains_pillars[:voxels.shape[0]] = 1

        pillars = np.zeros((self.n_pillars, self.points_per_pillar, 9))

        # 3.fill pillars with information according to paper (x,y,z,i,dx_pillar,dy_pillar,dcenterpillar_xyz) -> 9 values
        pillars[:voxels.shape[0], :, :4] = voxels
        pillars[:voxels.shape[0], :, 4:5] = voxels[..., 0:1] - ((np.expand_dims(coors[..., 1:2], 1)) * self.xy_voxel_size[0] + self.x_offset)
        pillars[:voxels.shape[0], :, 5:6] = voxels[..., 1:2] - ((np.expand_dims(coors[..., 2:], 1)) * self.xy_voxel_size[1] + self.y_offset)
        pillar_means = voxels[..., :3].sum(1) / np.expand_dims(num_points_per_voxel, -1)
        pillars[:voxels.shape[0], :, 6:] = voxels[..., :3] - np.expand_dims(pillar_means, 1)

        # #4.Before normalizing the pillars, we calculate the pillar's location on img (W,H, 0->1),
        # # This information will be used to index the rgb's deep learned features and concatenated with deep learned pillars
        # pillar_loc_xyz = pillar_means
        # pillar_img_pts = calib.project_velo_to_image(pillar_loc_xyz) #same n_samples as voxels
        # pillar_img_filter = np.zeros((self.n_pillars))
        # img_pnts_filter = (pillar_img_pts[:,0]>=0)&(pillar_img_pts[:,0]<=W)&(pillar_img_pts[:,1]>=0)&(pillar_img_pts[:,1]<=H)
        # pillar_img_filter[:voxels.shape[0]][img_pnts_filter] = 1
        # # pillar_img_pts = pillar_img_pts[img_pnts_filter]
        # pillar_img_pts = pillar_img_pts/np.array([W,H])
        # pillar_img_pts2 = np.zeros((self.n_pillars,2))
        # pillar_img_pts2[:voxels.shape[0]] = pillar_img_pts

        # 4.Before normalizing the pillars, we calculate the pillar's location on img (W,H, 0->1),
        # This information will be used to index the rgb's deep learned features and concatenated with deep learned pillars
        contains_rgb = np.zeros(self.n_pillars)
        rgb_coors = np.zeros((self.n_pillars, 3))
        pillar_img_pts2 = np.zeros((self.n_pillars, 2))
        pillar_loc_xyz = pillar_means
        pillar_img_pts = calib.project_velo_to_image(pillar_loc_xyz)                                                      # project pillar means to rgb image
        img_pnts_filter = (pillar_img_pts[:, 0] > 0) & (pillar_img_pts[:, 0] < W) & (pillar_img_pts[:, 1] > 0) & (pillar_img_pts[:, 1] < H)  # include only pillar means in photo
        n_rgb = np.sum(img_pnts_filter)
        contains_rgb[:n_rgb] = 1                                                                                           # set bool index to say how many rgb points is in the voxel
        rgb_coors[:n_rgb] = coord[:voxels.shape[0]][img_pnts_filter]                                                      # indicate which rgb point corresponds to which pillar
        pillar_img_pts2[:n_rgb] = pillar_img_pts[img_pnts_filter]                                                         # scatter the filtered points back to tensor
        pillar_img_pts2 = pillar_img_pts2 / np.array([W, H])                                                                 # normalize to between 0,1, represents position (x,y) on image

        # 5.Normalize pillars for faster convergence
        pillars[..., 0] = ((pillars[..., 0] - self.xyz_range[0]) / (self.xyz_range[3] - self.xyz_range[0]))
        pillars[..., 1] = ((pillars[..., 1] - self.xyz_range[1]) / (self.xyz_range[4] - self.xyz_range[1]))
        pillars[..., 2] = ((pillars[..., 2] - self.xyz_range[2]) / (self.xyz_range[5] - self.xyz_range[2]))
        pillars[..., 4] = pillars[..., 4] / (self.xyz_range[3] - self.xyz_range[0])
        pillars[..., 5] = pillars[..., 5] / (self.xyz_range[4] - self.xyz_range[1])
        pillars[..., 6] = pillars[..., 6] / (self.xyz_range[3] - self.xyz_range[0])
        pillars[..., 7] = pillars[..., 7] / (self.xyz_range[4] - self.xyz_range[1])
        pillars[..., 8] = pillars[..., 8] / (self.xyz_range[5] - self.xyz_range[2])

        label = pd.read_csv(label_file, header=None, sep=' ')
        label.columns = self.col_names
        df = label[['type', 'z', 'x', 'y', 'l', 'w', 'h', 'yaw']]  # Camera Coord
        df.columns = ['type', 'x', 'y', 'z', 'l', 'w', 'h', 'yaw']  # LiDAR Coord
        df['y'] = (-1 * df['y']).copy(deep=True)
        df = df[df['type'] != 'DontCare']
        xy_filter = (df['x'].values <= self.xyz_range[3]) & (df['x'].values >= self.xyz_range[0]) & (df['y'].values <= self.xyz_range[4]) & (df['y'].values >= self.xyz_range[1])
        df = df[xy_filter]
        classes_int = [self.classes[l] for l in df['type'].values]

        pts = df.iloc[:, 1:].values
        boxes = torch.as_tensor(pts).view(-1, 7).float()

        # ===============================[NORMALIZE ALL TARGETS TO BETWEEN 0-1]==================================================#
        # boxes[...,0:1] =  torch.clamp( ((boxes[:,0:1] - self.xyz_range[0])  / (self.xyz_range[3]-self.xyz_range[0])) , 0 ,1 ) #x
        # boxes[...,1:2] =  torch.clamp( ((boxes[:,1:2] - self.xyz_range[1])  / (self.xyz_range[4]-self.xyz_range[1])) , 0 ,1 ) #y
        # boxes[...,2:3] =  torch.clamp( ((boxes[:,2:3] - self.xyz_range[2])  / (self.xyz_range[5]-self.xyz_range[2])) , 0 ,1 ) #z
        # boxes[...,3:4] =  boxes[...,3:4]/(self.xyz_range[3]-self.xyz_range[0]) #l
        # boxes[...,4:5] =  boxes[...,4:5]/(self.xyz_range[4]-self.xyz_range[1]) #w
        # boxes[...,5:6] =  boxes[...,5:6]/(self.xyz_range[5]-self.xyz_range[2]) #h
        # boxes[...,6:]  =  (boxes[...,6:]/np.pi+1)/2 #normalize -pi to pi -> 0 to 1 #rot

        outputs = {}
        outputs['boxes'] = boxes
        outputs['labels'] = torch.tensor(classes_int, dtype=torch.int).long()
        outputs['idx'] = torch.as_tensor([idx])

        img = cv2.resize(img, (W // 2, H // 2)) / 255.0
        img = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1)

        if self.return_calib:
            return img, (pillars, coord, contains_pillars), (pillar_img_pts2, rgb_coors, contains_rgb), outputs, calib
        else:
            return img, (pillars, coord, contains_pillars), (pillar_img_pts2, rgb_coors, contains_rgb), outputs

    def __len__(self):
        return len(self.pc_filenames_train)


def KITTI_collate_fn(data):
    """
        Collate Function for NOAA so that Input/Targets can be concatenated
    """
    imgs, pillar_batch, rgb_batch, outputs = zip(*data, strict=False)

    imgs = torch.cat([im.unsqueeze(0) for im in imgs])
    pillars = torch.stack([torch.as_tensor(p[0]) for p in pillar_batch])
    coords = torch.stack([torch.as_tensor(p[1]) for p in pillar_batch])
    contains_pillars = torch.stack([torch.as_tensor(p[2]) for p in pillar_batch])

    rgbpillars = torch.stack([torch.as_tensor(p[0]) for p in rgb_batch])
    rgbcoords = torch.stack([torch.as_tensor(p[1]) for p in rgb_batch])
    rgbcontains_pillars = torch.stack([torch.as_tensor(p[2]) for p in rgb_batch])

    return imgs, [pillars, coords, contains_pillars], [rgbpillars, rgbcoords, rgbcontains_pillars], [x for x in outputs]


def KITTI_collate_fn_Wcalib(data):
    """
        Collate Function for NOAA so that Input/Targets can be concatenated
    """
    imgs, pillar_batch, rgb_batch, outputs, calibs = zip(*data, strict=False)

    imgs = torch.cat([im.unsqueeze(0) for im in imgs])
    pillars = torch.stack([torch.as_tensor(p[0]) for p in pillar_batch])
    coords = torch.stack([torch.as_tensor(p[1]) for p in pillar_batch])
    contains_pillars = torch.stack([torch.as_tensor(p[2]) for p in pillar_batch])

    rgbpillars = torch.stack([torch.as_tensor(p[0]) for p in rgb_batch])
    rgbcoords = torch.stack([torch.as_tensor(p[1]) for p in rgb_batch])
    rgbcontains_pillars = torch.stack([torch.as_tensor(p[2]) for p in rgb_batch])

    return imgs, [pillars, coords, contains_pillars], [rgbpillars, rgbcoords, rgbcontains_pillars], [x for x in outputs], [x for x in calibs]

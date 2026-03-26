import torch
from torch.utils.data import Dataset
from pvnet.utils.pcd_preprocess import point_set_to_coord_feats, aggregate_pcds, load_poses
from pvnet.utils.pcd_transforms import *
from pvnet.utils.data_map import learning_map
from pvnet.utils.collations import point_set_to_sparse
from natsort import natsorted
import os
import numpy as np
import yaml
import open3d as o3d

import warnings
import glob

warnings.filterwarnings('ignore')

#################################################
################## Data loader ##################
#################################################

class TemporalKITTISet(Dataset):
    def __init__(self, data_dir, seqs, split, resolution, num_points, max_range, max_bound,
            min_bound, grid_sizes, dataset_norm=False, std_axis_norm=False):
        super().__init__()
        self.data_dir = data_dir

        self.n_clusters = 50
        self.resolution = resolution
        self.num_points = num_points
        self.max_range = max_range
        self.max_bound = max_bound
        self.min_bound = min_bound
        self.grid_sizes = grid_sizes

        self.split = split
        self.seqs = seqs
        self.cache_maps = {}

        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath_list()
        self.data_stats = {'mean': None, 'std': None}

        if os.path.isfile(f'utils/data_stats_range_{int(self.max_range)}m.yml') and dataset_norm:
            stats = yaml.safe_load(open(f'utils/data_stats_range_{int(self.max_range)}m.yml'))
            data_mean = np.array([stats['mean_axis']['x'], stats['mean_axis']['y'], stats['mean_axis']['z']])
            if std_axis_norm:
                data_std = np.array([stats['std_axis']['x'], stats['std_axis']['y'], stats['std_axis']['z']])
            else:
                data_std = np.array([stats['std'], stats['std'], stats['std']])
            self.data_stats = {
                'mean': torch.tensor(data_mean),
                'std': torch.tensor(data_std)
            }

        self.nr_data = len(self.points_datapath)

        print('The size of %s data is %d'%(self.split,len(self.points_datapath)))

    def datapath_list(self):
        self.points_datapath = []
        self.seq_poses = []
        
        for seq in self.seqs:
            point_seq_path = os.path.join(self.data_dir, 'dataset', 'sequences', seq)
            voxel_file_path = natsorted(glob.glob(os.path.join(point_seq_path, 'voxels', '*.bin')))
            point_seq_bin = natsorted(os.listdir(os.path.join(point_seq_path, 'velodyne')))
            poses = load_poses(os.path.join(point_seq_path, 'calib.txt'), os.path.join(point_seq_path, 'poses.txt'))
            p_full = np.load(f'{point_seq_path}/clean_map.npy') if self.split != 'test' else np.array([[1,0,0],[0,1,0],[0,0,1]])
            self.cache_maps[seq] = p_full
            
            for file_index in voxel_file_path:
                file_num = int(file_index.split("/")[-1][:-4])
                self.points_datapath.append(os.path.join(point_seq_path, 'velodyne', point_seq_bin[file_num]))
                self.seq_poses.append(poses[file_num])
        
        if self.split != 'train':
            self.points_datapath = self.points_datapath[:40]   # val: 40, test: 200
            self.seq_poses = self.seq_poses[:40]
            
    def transforms(self, points):
        points = np.expand_dims(points, axis=0)
        points[:,:,:3] = rotate_point_cloud(points[:,:,:3])
        points[:,:,:3] = rotate_perturbation_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_scale_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_flip_point_cloud(points[:,:,:3])

        return np.squeeze(points, axis=0)
        
    def __getitem__(self, index):
        
        seq_num = self.points_datapath[index].split('/')[-3]
        fname = self.points_datapath[index].split('/')[-1].split('.')[0]

        p_part = np.fromfile(self.points_datapath[index], dtype=np.float32)
        p_part = p_part.reshape((-1,4))[:,:3]

        if self.split != 'test':
            label_file = self.points_datapath[index].replace('velodyne', 'labels').replace('.bin', '.label')
            l_set = np.fromfile(label_file, dtype=np.uint32)
            l_set = l_set.reshape((-1))
            l_set = l_set & 0xFFFF
            static_idx = (l_set < 252) & (l_set > 1)
            p_part = p_part[static_idx]
        dist_part = np.sum(p_part**2, -1)**.5
        p_part = p_part[(dist_part < self.max_range) & (dist_part > 3.5)]
        p_part = p_part[p_part[:,2] > -4.]
        pose = self.seq_poses[index]

        p_map = self.cache_maps[seq_num]

        if self.split != 'test':
            trans = pose[:-1,-1]
            dist_full = np.sum((p_map - trans)**2, -1)**.5
            p_full = p_map[dist_full < self.max_range]
            p_full = np.concatenate((p_full, np.ones((len(p_full),1))), axis=-1)
            p_full = (p_full @ np.linalg.inv(pose).T)[:,:3]
            p_full = p_full[p_full[:,2] > -4.]
        else:
            p_full = p_part

        if self.split == 'train':
            p_concat = np.concatenate((p_full, p_part), axis=0)
            p_concat = self.transforms(p_concat)

            p_full = p_concat[:-len(p_part)]
            p_part = p_concat[-len(p_part):]
        
        # patial pcd has 1/10 of the complete pcd size
        n_part = int(self.num_points / 10.)

        concat_part = np.ceil(n_part / p_part.shape[0]) 
        p_part = p_part.repeat(concat_part, 0)
        
        pcd_part = o3d.geometry.PointCloud()
        pcd_part.points = o3d.utility.Vector3dVector(p_part)
        viewpoint_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_part, voxel_size=10.)
        pcd_part = pcd_part.farthest_point_down_sample(n_part)
        p_part = np.array(pcd_part.points)
                
        in_viewpoint = viewpoint_grid.check_if_included(o3d.utility.Vector3dVector(p_full))
        p_full = p_full[in_viewpoint] 
        concat_full = np.ceil(self.num_points / p_full.shape[0])

        p_full = p_full[torch.randperm(p_full.shape[0])]
        p_full = p_full.repeat(concat_full, 0)[:self.num_points]
        
        p_part = torch.tensor(p_part)
        p_full = torch.tensor(p_full)
        
        filename = self.points_datapath[index]

        return [p_full, p_part, filename]    
        
    def __len__(self):
        return self.nr_data

##################################################################################################

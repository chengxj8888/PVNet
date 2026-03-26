import time
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from module.voxel_fea_precess import voxel_fea
from module.voxel_fea_extraction import Voxel_fea_extract
from module.shared_mlp import SharedMLP
import nearest_neighbors as nearest_neighbors
import spconv 
import open3d as o3d 
from pykeops.torch import LazyTensor
from module.utils import get_knn_pts, index_points
from einops import rearrange, repeat
from module.voxelization import Voxelization
from module.utils import *

__all__ = ['UNetDiff']

def align_pnt(pnt, voxel_size, pnt_range):
    device = pnt.device
    voxel_size = torch.Tensor(voxel_size).to(device)
    pnt_min_range = torch.Tensor(pnt_range).to(device)   
    pnt[:, 1:] = (pnt[:, 1:].float() + 0.5) * voxel_size + pnt_min_range
    return pnt.float()

def index_feat(feature, index):
    device = index.device
    N, K = index.shape
    mask = None
    
    if K > 1:
        group_first = index[:, 0].view((N, 1)).repeat([1, K]).to(device)        
        mask = index == 0                 
        index[mask] = group_first[mask]   

    flat_index = index.reshape((N * K,))        
    
    selected_feat = feature[flat_index, ]       
    
    if K > 1:
        selected_feat = selected_feat.reshape((N, K, -1))       
    else:
        selected_feat = selected_feat.reshape((N, -1))
    return selected_feat, mask

def relation_position(group_xyz, center_xyz):
    K = group_xyz.shape[1]      
    tile_center = center_xyz.unsqueeze(1).repeat([1, K, 1])     
    offset = group_xyz - tile_center   
    dist = torch.norm(offset, p=None, dim=-1, keepdim=True)     
    relation = torch.cat([offset, tile_center, group_xyz, dist], -1)    
    
    return relation


class UNetDiff(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        in_channels = kwargs.get('in_channels', 3)
        cs = [32, 64, 128, 256, 128, 96]
        cs = [int(cr * x) for x in cs] 
        self.embed_dim = cs[-1]
        self.run_up = kwargs.get('run_up', True)
        self.D = kwargs.get('D', 3)
        self.channels = cs
        self.k = 16     # default: 16

        self.max_bound = np.asarray(kwargs['data']['max_volume_space'])       
        self.min_bound = np.asarray(kwargs['data']['min_volume_space'])       
        self.grid_sizes = kwargs['data']['grid_sizes']                   
        self.point_range = kwargs['data']['point_cloud_range']           
        self.fea_dim = kwargs['model']['fea_dim']                        
        self.out_pt_fea_dim = kwargs['model']['out_fea_dim']             
        self.init_size = kwargs['model']['init_size']                    
        self.feat_relation = True
        
        self.voxelize = Voxelization(max_bound=self.max_bound, min_bound=self.min_bound, cur_grid_size=self.grid_sizes[0])
        self.voxel_fea_precess = voxel_fea(fea_dim=self.init_size[1], out_pt_fea_dim=self.out_pt_fea_dim, fea_compre=self.init_size[0])
        self.voxel_fea_extract = Voxel_fea_extract(output_shape=self.grid_sizes[0], nclasses=cs[1], init_size=self.init_size[0])

        self.mlp_full = nn.Sequential(
            SharedMLP(self.fea_dim[0]+cs[-1]+3, self.init_size[1]),
            nn.Conv1d(self.init_size[1], self.init_size[1], kernel_size=1)
        )
        
        self.mlp_pt = nn.Sequential(
            SharedMLP(in_channels, self.init_size[0]),
            nn.Conv1d(self.init_size[0], self.init_size[1], kernel_size=1)
        )

        self.relation_w = nn.Sequential(
                nn.Conv1d(cs[1]+cs[1]+cs[1]+10, cs[2], kernel_size=1),
                nn.BatchNorm1d(cs[2]),
                nn.LeakyReLU(0.2),
                nn.Conv1d(cs[2], cs[1], kernel_size=1)
            ) 
        
        self.mlp_devoxel = nn.Sequential(
            SharedMLP(cs[1], cs[1]),
            nn.Conv1d(cs[1], cs[1], kernel_size=1)
        )  

        self.latent = nn.Sequential(
            nn.Linear(self.out_pt_fea_dim, cs[2]),              
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(cs[2], cs[1]),
        )

        self.final_point = nn.Sequential(
            nn.Conv1d(cs[1], 20, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(20, 3, kernel_size=1)
            )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def match_part_to_full(self, full_coord, part_coord, part_feat):
        full_c = full_coord.clone().reshape(-1,3).float()     
        part_c = part_coord.clone().reshape(-1,3).float()    
        
        f_coord = LazyTensor(full_c[:,None,:])
        p_coord = LazyTensor(part_c[None,:,:])

        dist_fp = ((f_coord - p_coord)**2).sum(-1)
        match_index= dist_fp.argKmin(1,dim=1)[:,0]
        
        part_feat = part_feat.reshape(-1, part_feat.shape[2])       

        match_feat = part_feat[match_index].reshape(full_coord.shape[0], full_coord.shape[1], -1)  
        
        return match_feat
 
    def get_timestep_embedding(self, timesteps):
        assert len(timesteps.shape) == 1 

        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(torch.device('cuda'))
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if self.embed_dim % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        assert emb.shape == torch.Size([timesteps.shape[0], self.embed_dim])
        return emb

    def forward(self, x_full, x_full_coord, part_feat, part_coord, t):
        t_emb = self.get_timestep_embedding(t)       
        t_emb = t_emb.unsqueeze(1).repeat(1, x_full.shape[1], 1)    

        batch_size = x_full.shape[0]
        num_points = x_full.shape[1]
        
        coords = x_full_coord.contiguous()

        pt_feats_list = []

        point_devoxel_list = []

        grid_full_index, x_full_feat = self.voxelize(coords)    # voxelization
        
        x_full_feat = torch.cat((x_full_feat, x_full, t_emb), dim=2)      

        x_full_feat = self.mlp_full(x_full_feat.permute(0,2,1)).permute(0,2,1)     
        
        voxel_coords_uni, voxel_features_3d = self.voxel_fea_precess(x_full_feat, grid_full_index)

        voxel_coords, voxel_feats = self.voxel_fea_extract(voxel_features_3d, voxel_coords_uni, batch_size)  
        
        pt_list = []
        for i_batch in range(batch_size):
            pt_list.append(F.pad(coords[i_batch], (1, 0), 'constant', value=i_batch))
        
        point_indices = torch.cat(pt_list, dim=0)       
            
        pt_feats = self.mlp_pt(x_full.permute(0,2,1)).permute(0,2,1)           
        point_feats = pt_feats.reshape(pt_feats.shape[0]*pt_feats.shape[1], pt_feats.shape[2])     
        
        # compute the coordinates of voxel center points
        voxel_size = (self.max_bound - self.min_bound) / (np.asarray(self.grid_sizes[0]) - 1)    
        voxel_points_coord = align_pnt(voxel_coords, voxel_size, self.min_bound)       
        
        indices_points = torch.zeros(point_indices.shape[0], self.k)        
        
        offset_center = 0
        offset_query = 0
        voxel_index = 0

        for b in range(batch_size):   
            query_points = voxel_points_coord[voxel_points_coord[:, 0] == b][:, 1:].unsqueeze(0).to('cpu').data.numpy()  
            center_points = point_indices[point_indices[:,0]==b][:, 1:].unsqueeze(0).to('cpu').data.numpy()         
            indexs = nearest_neighbors.knn_batch(query_points, center_points, self.k, omp=True)     
            assert len(indexs[0]) == len(center_points[0])
            
            num_center_point = center_points.shape[1]    # 32768
            num_query_point = query_points.shape[1]      # 180000
            
            indices_points[offset_center:num_center_point+offset_center] = torch.Tensor(indexs+offset_query).squeeze(0)

            offset_center += num_center_point
            offset_query += num_query_point

        indices_points = indices_points.long().to(x_full.device)     

        voxel_feat_combin = torch.cat([voxel_points_coord[:, 1:].float(), voxel_feats], 1)      
        group, _ = index_feat(voxel_feat_combin, indices_points) 
        group_xyz, group_features = group[:, :, :3], group[:, :, 3:] 

        relation = relation_position(group_xyz, point_indices[:, 1:].float()) 
        
        match_part_feat = self.match_part_to_full(coords, part_coord, part_feat)     

        match_feat = match_part_feat.reshape(match_part_feat.shape[0]*match_part_feat.shape[1], match_part_feat.shape[2])       

        match_feat = self.latent(match_feat)    # (2*360000, 64)

        if self.feat_relation:
            relation = torch.cat([relation, group_features, point_feats.unsqueeze(1).repeat((1, self.k, 1)), match_feat.unsqueeze(1).repeat((1, self.k, 1))], -1)   # (360000, 16, 64+64+64+10)
        
        group_w = self.relation_w(relation.permute(0,2,1))    
        group_features = group_features.permute(0, 2, 1)      
        
        group_features *= group_w
        updated_features = torch.mean(group_features, 2)            # average
        updated_features = updated_features.reshape(batch_size, num_points, -1).permute(0,2,1)      
        
        point_devoxel_feat = self.mlp_devoxel(updated_features)    
        
        out = self.final_point(point_devoxel_feat)      
        
        return out.permute(0,2,1)


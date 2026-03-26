# -*- coding:utf-8 -*-
# author: Xinge

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

class voxel_fea(nn.Module):
    def __init__(self, fea_dim=3, out_pt_fea_dim=64, fea_compre=None):
        super(voxel_fea, self).__init__()
        
        self.PPmodel = nn.Sequential(
            nn.Linear(fea_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, out_pt_fea_dim)
        )
        
        self.fea_compre = fea_compre
        self.shuffled = False
        
        # point feature compression
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(out_pt_fea_dim, self.fea_compre),
                nn.ReLU())
        
    def forward(self, pt_fea, xy_ind):
        cur_dev = pt_fea[0].get_device()
        
        pt_fea_list = []
        xy_ind_list = []

        for b in range(pt_fea.shape[0]):
            pt_fea_list.append(pt_fea[b])
            xy_ind_list.append(xy_ind[b])
        
        # concate everything
        cat_pt_ind = []
        for i_batch in range(len(xy_ind_list)):
            cat_pt_ind.append(F.pad(xy_ind_list[i_batch], (1, 0), 'constant', value=i_batch))  
        
        cat_pt_fea = torch.cat(pt_fea_list, dim=0)      
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)       
        pt_num = cat_pt_ind.shape[0]
        
        # shuffle the data 打乱了原始的点云所属voxel的顺序
        if self.shuffled:
            shuffled_ind = torch.randperm(pt_num, device=cur_dev)
            cat_pt_fea = cat_pt_fea[shuffled_ind, :]
            cat_pt_ind = cat_pt_ind[shuffled_ind, :]

        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)     
        
        processed_cat_pt_fea = self.PPmodel(cat_pt_fea)     
        pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]    
        
        if self.fea_compre:     
            processed_pooled_data = self.fea_compression(pooled_data)   
        else:
            processed_pooled_data = pooled_data
        
        return unq, processed_pooled_data
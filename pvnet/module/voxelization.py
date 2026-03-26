import torch
import torch.nn as nn
import numpy as np

__all__ = ['Voxelization']

class Voxelization(nn.Module):
    def __init__(self, max_bound, min_bound, cur_grid_size):
        super().__init__()
        self.max_bound = max_bound
        self.min_bound = min_bound
        self.cur_grid_size = cur_grid_size
    
    def forward(self, xyz):
        
        device = xyz.device
        max_bound = torch.from_numpy(np.asarray(self.max_bound)).float().to(device)
        min_bound = torch.from_numpy(np.asarray(self.min_bound)).float().to(device)
        
        crop_range = max_bound - min_bound      
        
        cur_grid_size = torch.from_numpy(np.asarray(self.cur_grid_size)).float().to(device)
        
        intervals = crop_range / (cur_grid_size - 1)        

        if (intervals == 0).any(): print("Zero interval!")
                
        grid_ind = (torch.floor((torch.clip(xyz, min_bound, max_bound) - min_bound) / intervals)).int()    
        
        # center data on each voxel
        voxel_centers = (grid_ind.float() + 0.5) * intervals + min_bound     
        
        return_xyz_delta = xyz - voxel_centers    
        
        grid_ind_reshape = grid_ind.reshape(-1)
        
        return grid_ind, return_xyz_delta.float()


    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')

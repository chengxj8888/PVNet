import torch
import math
from einops import rearrange
from pvnet.lib.pointops.functions import pointops
import logging
import os
import numpy as np
import random
from torch.autograd import grad
from einops import rearrange, repeat

def copy_source(file, output_dir):
    copyfile(file, os.path.join(output_dir, os.path.basename(file)))

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def index_points(pts, idx):
    """
    Input:
        pts: input points data, [B, C, N]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, C, S, [K]]
    """ 
    batch_size = idx.shape[0]
    sample_num = idx.shape[1]
    fdim = pts.shape[1]     
    reshape = False
    if len(idx.shape) == 3:
        reshape = True
        idx = idx.reshape(batch_size, -1)  
    
    # (b, c, (s k))
    res = torch.gather(pts, 2, idx[:, None].repeat(1, fdim, 1))
    if reshape:
        res = rearrange(res, 'b c (s k) -> b c s k', s=sample_num)   
    
    return res


def FPS(pts, fps_pts_num):
    # input: (b, 3, n)
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    # (b, fps_pts_num)
    sample_idx = pointops.furthestsampling(pts_trans, fps_pts_num).long()
    # (b, 3, fps_pts_num)
    sample_pts = index_points(pts, sample_idx)
    
    return sample_pts


def get_knn_pts(k, pts, center_pts, return_idx=False):
    # (b, n, 3)
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()   
    # (b, m, 3)
    center_pts_trans = rearrange(center_pts, 'b c m -> b m c').contiguous()    
    knn_idx = pointops.knnquery_heap(k, pts_trans, center_pts_trans).long()    # (b, m, k)
    # (b, 3, m, k)
    knn_pts = index_points(pts, knn_idx)     
    
    if return_idx == False:
        return knn_pts
    else:
        return knn_pts, knn_idx

def get_knn_gt_pts(k, pts, center_pts, return_idx=False):
    # (b, n, 3)
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()  
    # (b, m, 3)
    center_pts_trans = rearrange(center_pts, 'b c m -> b m c').contiguous()     
    # (b, m, k)
    knn_idx = pointops.knnquery_heap(k, pts_trans, center_pts_trans).long()     
    # (b, 3, m, k)
    knn_pts = index_points(pts, knn_idx)    
    
    if return_idx == False:
        return knn_pts
    else:
        return knn_pts, knn_idx

def interpolate_feature(support_pts, center_pts, support_feat):
    support_pts = support_pts.permute(0,2,1)   
    center_pts = center_pts.permute(0,2,1)      
    support_feat = support_feat.permute(0,2,1)  
    
    k = 8 
    # interpolation: knn_pts: (b, 3, n, k),  knn_idx: (b, n, k)
    knn_pts, knn_idx = get_knn_pts(k, support_pts, center_pts, return_idx=True)
    
    # dist 
    repeat_center_pts = repeat(center_pts, 'b c n -> b c n k', k=k)   
    
    # (b, n, k)
    dist = torch.norm(knn_pts - repeat_center_pts, p=2, dim=1)      
    dist_recip = 1.0 / (dist + 1e-8)   
    norm = torch.sum(dist_recip, dim=2, keepdim=True)   
    
    # (b, n, k)
    weight = dist_recip / norm          
    # (b, c, n, k)
    knn_feat = index_points(support_feat, knn_idx)      
    # (b, c, n, k)
    interpolated_feat = knn_feat * weight.unsqueeze(1)  
    # (b, c, n)
    interpolated_feat = torch.sum(interpolated_feat, dim=-1)    
    return interpolated_feat

def midpoint_interpolate(args, sparse_pts):
    # sparse_pts: input points (b, 3, n)
    pts_num = sparse_pts.shape[-1]
    up_pts_num = int(pts_num * args.up_rate)    
    k = int(2 * args.up_rate)  
    # (b, 3, n, k)
    knn_pts = get_knn_pts(k, sparse_pts, sparse_pts)
    # (b, 3, n, k)
    repeat_pts = repeat(sparse_pts, 'b c n -> b c n k', k=k)
    # (b, 3, n, k)
    mid_pts = (knn_pts + repeat_pts) / 2.0
    # (b, 3, (n k))
    mid_pts = rearrange(mid_pts, 'b c n k -> b c (n k)')
    # note that interpolated_pts already contain sparse_pts
    interpolated_pts = mid_pts  # (B, 3, 2048)
    interpolated_pts = FPS(interpolated_pts, up_pts_num)    

    return interpolated_pts

def normalize_point_cloud(input, centroid=None, furthest_distance=None):
    # input: (b, 3, n) tensor
    if centroid is None:
        # (b, 3, 1)
        centroid = torch.mean(input, dim=-1, keepdim=True)
    # (b, 3, n)
    input = input - centroid
    if furthest_distance is None:
        # (b, 3, n) -> (b, 1, n) -> (b, 1, 1)
        furthest_distance = torch.max(torch.norm(input, p=2, dim=1, keepdim=True), dim=-1, keepdim=True)[0]
    input = input / furthest_distance

    return input, centroid, furthest_distance


def add_noise(pts, sigma, clamp):
    # input: (b, 3, n)
    assert (clamp > 0)
    jittered_data = torch.clamp(sigma * torch.randn_like(pts), -1 * clamp, clamp).cuda()
    jittered_data += pts

    return jittered_data

# generate patch for test
def extract_knn_patch(k, pts, center_pts):
    # input : (b, 3, n)
    # (n, 3)
    pts_trans = rearrange(pts.squeeze(0), 'c n -> n c').contiguous()
    pts_np = pts_trans.detach().cpu().numpy()
    # (m, 3)
    center_pts_trans = rearrange(center_pts.squeeze(0), 'c m -> m c').contiguous()
    center_pts_np = center_pts_trans.detach().cpu().numpy()
    knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_search.fit(pts_np)
    # (m, k)
    knn_idx = knn_search.kneighbors(center_pts_np, return_distance=False)
    # (m, k, 3)
    patches = np.take(pts_np, knn_idx, axis=0)
    patches = torch.from_numpy(patches).float().cuda()
    # (m, 3, k)
    patches = rearrange(patches, 'm k c -> m c k').contiguous()

    return patches


def get_logger(name, log_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')
    # output to console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # output to log file
    log_name = name + '_log.txt'
    file_handler = logging.FileHandler(os.path.join(log_dir, log_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_query_points(input_pts, args):
    query_pts = input_pts + (torch.randn_like(input_pts) * args.local_sigma)

    return query_pts


def reset_model_args(train_args, model_args):
    for arg in vars(train_args):
        setattr(model_args, arg, getattr(train_args, arg))


def get_output_dir(prefix, exp_id):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join(prefix, 'output/' + exp_id, t)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir
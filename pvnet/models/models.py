import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from pvnet.utils.scheduling import beta_func
from tqdm import tqdm
from os import makedirs, path

from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.core.datamodule import LightningDataModule
from pvnet.utils.collations import *
from pvnet.utils.metrics import ChamferDistance, PrecisionRecall
from diffusers import DPMSolverMultistepScheduler
from .part_feats_extraction import Part_feat_Extraction
from .diffusionUNet import UNetDiff
from pvnet.utils.metrics import P2C

class DiffusionPoints(LightningModule):
    def __init__(self, hparams:dict, data_module: LightningDataModule = None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module

        # alphas and betas
        if self.hparams['diff']['beta_func'] == 'cosine':
            self.betas = beta_func[self.hparams['diff']['beta_func']](self.hparams['diff']['t_steps'])
        else:
            self.betas = beta_func[self.hparams['diff']['beta_func']](
                    self.hparams['diff']['t_steps'],
                    self.hparams['diff']['beta_start'],
                    self.hparams['diff']['beta_end'],
            )

        self.t_steps = self.hparams['diff']['t_steps']
        self.s_steps = self.hparams['diff']['s_steps']
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.tensor(
            np.cumprod(self.alphas, axis=0), dtype=torch.float32, device=torch.device('cuda')
        )

        self.alphas_cumprod_prev = torch.tensor(
            np.append(1., self.alphas_cumprod[:-1].cpu().numpy()), dtype=torch.float32, device=torch.device('cuda')
        )

        self.betas = torch.tensor(self.betas, device=torch.device('cuda'))
        self.alphas = torch.tensor(self.alphas, device=torch.device('cuda'))

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod) 
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1.)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.sqrt_posterior_variance = torch.sqrt(self.posterior_variance)
        self.posterior_log_var = torch.log(
            torch.max(self.posterior_variance, 1e-20 * torch.ones_like(self.posterior_variance))
        )

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        
        # for fast sampling
        self.dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.t_steps,
                beta_start=self.hparams['diff']['beta_start'],
                beta_end=self.hparams['diff']['beta_end'],
                beta_schedule='linear',
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.dpm_scheduler.set_timesteps(self.s_steps)
        self.scheduler_to_cuda()
        
        self.partial_enc = Part_feat_Extraction(in_channels=3)
        self.model = UNetDiff(in_channels=3, **self.hparams)

        self.chamfer_distance = ChamferDistance()
        self.precision_recall = PrecisionRecall(self.hparams['data']['resolution'],2*self.hparams['data']['resolution'],100)

        self.metric_val = P2C()

        self.w_uncond = self.hparams['train']['uncond_w']

    def scheduler_to_cuda(self):
        self.dpm_scheduler.timesteps = self.dpm_scheduler.timesteps.cuda()
        self.dpm_scheduler.betas = self.dpm_scheduler.betas.cuda()
        self.dpm_scheduler.alphas = self.dpm_scheduler.alphas.cuda()
        self.dpm_scheduler.alphas_cumprod = self.dpm_scheduler.alphas_cumprod.cuda()
        self.dpm_scheduler.alpha_t = self.dpm_scheduler.alpha_t.cuda()
        self.dpm_scheduler.sigma_t = self.dpm_scheduler.sigma_t.cuda()
        self.dpm_scheduler.lambda_t = self.dpm_scheduler.lambda_t.cuda()
        self.dpm_scheduler.sigmas = self.dpm_scheduler.sigmas.cuda()

    def q_sample(self, x, t, noise):
        return self.sqrt_alphas_cumprod[t][:,None,None].cuda() * x + \
                self.sqrt_one_minus_alphas_cumprod[t][:,None,None].cuda() * noise
    
    def classfree_forward(self, x_t, x_t_coord, x_part_con, x_part_uncond, t):       
        x_part_con = self.forward(x_t, x_t_coord, x_part_con, t)
        x_part_uncond = self.forward(x_t, x_t_coord, x_part_uncond, t)
        
        return x_part_uncond + self.w_uncond * (x_part_con - x_part_uncond)
    
    def visualize_step_t(self, x_t, gt_pts, pcd, pcd_mean, pcd_std, pidx=0):
        points = x_t.F.detach().cpu().numpy()
        points = points.reshape(gt_pts.shape[0],-1,3)
        obj_mean = pcd_mean[pidx][0].detach().cpu().numpy()
        points = np.concatenate((points[pidx], gt_pts[pidx]), axis=0)

        dist_pts = np.sqrt(np.sum((points - obj_mean)**2, axis=-1))
        dist_idx = dist_pts < self.hparams['data']['max_range']

        full_pcd = len(points) - len(gt_pts[pidx])
        print(f'\n[{dist_idx.sum() - full_pcd}|{dist_idx.shape[0] - full_pcd }] points inside margin...')

        pcd.points = o3d.utility.Vector3dVector(points[dist_idx])
        
        colors = np.ones((len(points), 3)) * .5
        colors[:len(gt_pts[0])] = [1.,.3,.3]
        colors[-len(gt_pts[0]):] = [.3,1.,.3]
        pcd.colors = o3d.utility.Vector3dVector(colors[dist_idx])

    def reset_partial_pcd(self, x_part_con, x_part_uncond, bs):
        x_part_con = x_part_con.detach()
        x_part_uncond = torch.zeros_like(x_part_con.reshape(bs,-1,3))

        return x_part_con, x_part_uncond

    def p_sample_loop(self, x_init, x_t, x_part_con, x_part_uncond, bs):
        self.scheduler_to_cuda()

        for t in tqdm(range(len(self.dpm_scheduler.timesteps))):
            t = torch.ones(bs).cuda().long() * self.dpm_scheduler.timesteps[t].cuda()
            
            noise_t = self.classfree_forward(x_t, x_init, x_part_con, x_part_uncond, t)    # (1, 18000, 3)
            input_noise = x_t.reshape(t.shape[0],-1,3) - x_init 
                        
            x_t = x_init + self.dpm_scheduler.step(noise_t, t[0], input_noise)['prev_sample']

            x_part_con, x_part_uncond = self.reset_partial_pcd(x_part_con, x_part_uncond, bs)
            torch.cuda.empty_cache()
        
        makedirs(f'{self.logger.log_dir}/generated_pcd/', exist_ok=True)
        
        return x_t
    
    def p_losses(self, y, noise):
        return F.mse_loss(y, noise)
    
    def _smooth_l1_loss(self, input, target, thres=1.0, reduction='none'):
        # type: (Tensor, Tensor) -> Tensor 
        t = torch.abs(input - target)  
        ret = torch.where(t < thres, 0.5 * t ** 2, t) 
        if reduction != 'none': 
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
        return ret 

    def forward(self, x_full, x_full_coord, x_part, t):
        part_feat = self.partial_enc(x_part) 
        out = self.model(x_full, x_full_coord, part_feat, x_part, t)
        torch.cuda.empty_cache()
        return out.reshape(t.shape[0], -1, 3)


    def training_step(self, batch:dict, batch_idx):
        # initial random noise
        torch.cuda.empty_cache()
        noise = torch.randn(batch['pcd_full'].shape, device=self.device)  # (b, npoints, 3)

        # sample step t
        t = torch.randint(0, self.t_steps, size=(batch['pcd_full'].shape[0],)).cuda()   # (b,)
        
        # sample q at step t    
        # we sample noise towards zero to then add to each point the noise (without normalizing the pcd)
        t_sample = self.q_sample(torch.zeros_like(batch['pcd_full']), t, noise)
        
        x_full = batch['pcd_full'] +  t_sample  # (b, n, 3)     # 生成的随机噪声+点云绝对坐标
                
        # for classifier-free guidance switch between conditional and unconditional training
        if torch.rand(1) > self.hparams['train']['uncond_prob'] or batch['pcd_full'].shape[0] == 1:
            x_part = batch['pcd_part']
        else:
            x_part = torch.zeros_like(batch['pcd_part'])
        
        x_full_coord = batch['pcd_full']
        
        denoise_t = self.forward(x_full, x_full_coord, x_part, t)
        loss_mse = self.p_losses(denoise_t, noise)
        
        loss_std = self._smooth_l1_loss(denoise_t.std(), 1.0)
        loss = loss_mse + loss_std*1.0
        
        std_noise = (denoise_t - noise)**2
        self.log('train/loss_mse', loss_mse)
        self.log('train/loss_std', loss_std)
        self.log('train/loss', loss)
        self.log('train/denoise_mean', denoise_t.mean(), on_step=True)
        self.log('train/denoise_std', denoise_t.std(), on_step=True)
        self.log('train/var', std_noise.var(), on_step=True)
        self.log('train/std', std_noise.std(), on_step=True)
        
        torch.cuda.empty_cache() 
        
        return loss
    
    def validation_step(self, batch:dict, batch_idx):
        self.model.eval()
        self.partial_enc.eval()
        with torch.no_grad():
            skip, output_paths, output_paths_gt, output_paths_part = self.valid_paths(batch['filename'])
            
            gt_pts = batch['pcd_full'].detach().cpu().numpy()

            # for inference we get the partial pcd and sample the noise around the partial
            x_init = batch['pcd_part'].repeat(1,10,1)
            x_full = x_init + torch.randn(x_init.shape, device=self.device)     
            x_part_con = batch['pcd_part']
            x_part_uncond = torch.zeros_like(batch['pcd_part'])
            
            bs = gt_pts.shape[0]

            x_gen_eval = self.p_sample_loop(x_init, x_full, x_part_con, x_part_uncond, bs)
                          
            CD, RCD, recon_RCD, match_RCD = self.metric_val.get_metrics(x_gen_eval, batch['pcd_part'])

            self.metric_val.update(CD, RCD, recon_RCD, match_RCD)

            for i in range(len(batch['pcd_full'])):
                pcd_pred = o3d.geometry.PointCloud()
                c_pred = x_gen_eval[i].cpu().detach().numpy()
                pcd_pred.points = o3d.utility.Vector3dVector(c_pred)

                pcd_gt = o3d.geometry.PointCloud()
                g_pred = batch['pcd_full'][i].cpu().detach().numpy()
                pcd_gt.points = o3d.utility.Vector3dVector(g_pred)
                
                self.chamfer_distance.update(pcd_gt, pcd_pred)
                self.precision_recall.update(pcd_gt, pcd_pred)

    def on_validation_epoch_end(self):
        cd_mean, cd_std = self.chamfer_distance.compute()
        pr, re, f1 = self.precision_recall.compute_auc()
        
        self.log('val/cd_mean', cd_mean, sync_dist=True)
        self.log('val/cd_std', cd_std, sync_dist=True)
        self.log('val/precision', pr, sync_dist=True)
        self.log('val/recall', re, sync_dist=True)
        self.log('val/fscore', f1, sync_dist=True)

        print(f'CD Mean: {cd_mean}\t CD Std: {cd_std}')
        print(f'Precision: {pr}\t Recall: {re}\t F-Score: {f1}')
        
        self.chamfer_distance.reset()
        self.precision_recall.reset()

        avg_CD, avg_RCD, avg_recon_RCD, avg_match_RCD = self.metric_val.compute()
        self.log('val/CD', avg_CD, sync_dist=True)
        self.log('val/RCD', avg_RCD, sync_dist=True)
        self.log('val/recon_RCD', avg_recon_RCD, sync_dist=True)
        self.log('val/match_RCD', avg_match_RCD, sync_dist=True)
        
        self.metric_val.reset()
        torch.cuda.empty_cache()

        return {'val/CD': avg_CD,'val/RCD': avg_RCD, 'val/recon_RCD': avg_recon_RCD, 'val/match_RCD': avg_match_RCD}

    def valid_paths(self, filenames):
        output_paths = []
        output_paths_gt = []
        output_paths_part = []
        skip = []

        for fname in filenames:
            seq_dir =  f'{self.logger.log_dir}/generated_pcd/{fname.split("/")[-3]}'
            ply_name = f'{fname.split("/")[-1].split(".")[0]}.ply'
            ply_name_gt = f'{fname.split("/")[-1].split(".")[0]}_gt.ply'
            ply_name_part = f'{fname.split("/")[-1].split(".")[0]}_part.ply'

            skip.append(path.isfile(f'{seq_dir}/{ply_name}'))
            makedirs(seq_dir, exist_ok=True)
            output_paths.append(f'{seq_dir}/{ply_name}')
            output_paths_gt.append(f'{seq_dir}/{ply_name_gt}')
            output_paths_part.append(f'{seq_dir}/{ply_name_part}')

        return np.all(skip), output_paths, output_paths_gt, output_paths_part

    def test_step(self, batch:dict, batch_idx):
        self.model.eval()
        self.partial_enc.eval()
        with torch.no_grad():
            skip, output_paths, output_paths_gt, output_paths_part = self.valid_paths(batch['filename'])

            if skip:
                print(f'Skipping generation from {output_paths[0]} to {output_paths[-1]}') 
                return {'test/cd_mean': 0., 'test/cd_std': 0., 'test/precision': 0., 'test/recall': 0., 'test/fscore': 0.}
            
            gt_pts = batch['pcd_full'].detach().cpu().numpy()

            noise_level = 0.1
            perturb_noise =  batch['pcd_part'] +  (torch.randn_like(batch['pcd_part']) * noise_level)
            k = 10      # upsampling rate
            x_init = perturb_noise.repeat(1,k,1)
            
            x_full = x_init + torch.randn(x_init.shape, device=self.device) 
            x_part_con = perturb_noise
            x_part_uncond = torch.zeros_like(perturb_noise)
            
            bs = gt_pts.shape[0]
            
            x_gen_eval = self.p_sample_loop(x_init, x_full, x_part_con, x_part_uncond, bs)
            
            CD, RCD, recon_RCD, match_RCD = self.metric_val.get_metrics(x_gen_eval, batch['pcd_part'])
            self.metric_val.update(CD, RCD, recon_RCD, match_RCD)
            
            for i in range(len(batch['pcd_full'])):
                pcd_pred = o3d.geometry.PointCloud()
                c_pred = x_gen_eval[i].cpu().detach().numpy()
                pcd_pred.points = o3d.utility.Vector3dVector(c_pred)

                pcd_gt = o3d.geometry.PointCloud()
                g_pred = batch['pcd_full'][i].cpu().detach().numpy()
                pcd_gt.points = o3d.utility.Vector3dVector(g_pred)

                self.chamfer_distance.update(pcd_gt, pcd_pred)
                self.precision_recall.update(pcd_gt, pcd_pred)
            
    def on_test_epoch_end(self):
        cd_mean, cd_std = self.chamfer_distance.compute()
        pr, re, f1 = self.precision_recall.compute_auc()
        print(f'CD Mean: {cd_mean}\t CD Std: {cd_std}')
        print(f'Precision: {pr}\t Recall: {re}\t F-Score: {f1}')

        self.log('test/cd_mean', cd_mean)
        self.log('test/cd_std', cd_std)
        self.log('test/precision', pr)
        self.log('test/recall', re)
        self.log('test/fscore', f1)
        
        self.chamfer_distance.reset()
        self.precision_recall.reset()
        
        avg_CD, avg_RCD, avg_recon_RCD, avg_match_RCD = self.metric_val.compute()
        print(f'avg_CD: {avg_CD}\t avg_CD: {avg_CD}')
        print(f'avg_recon_RCD: {avg_recon_RCD}\t avg_match_RCD: {avg_match_RCD}')

        self.log('test/CD', avg_CD, sync_dist=True)
        self.log('test/RCD', avg_RCD, sync_dist=True)
        self.log('test/recon_RCD', avg_recon_RCD, sync_dist=True)
        self.log('test/match_RCD', avg_match_RCD, sync_dist=True)
        
        self.metric_val.reset()
        torch.cuda.empty_cache()

        return {'test/cd_mean': cd_mean, 'test/cd_std': cd_std, 'test/precision': pr, 'test/recall': re, 'test/fscore': f1}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['train']['lr'], betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
        scheduler = {
            'scheduler': scheduler, # lr * 0.5
            'interval': 'epoch', # interval is epoch-wise
            'frequency': 5, # after 5 epochs
        }

        return [optimizer], [scheduler]
    
#######################################
# Modules
#######################################

import os
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from measure import compute_metrics
from config.ldct_models import get_ldct_model, FDD_MODEL_NAMES, TFA_MODEL_NAMES


def get_ckpt_path(test_ckpt):
    if Path(test_ckpt).is_file():        
        ckpt_path = Path(test_ckpt)
    else:
        raise FileNotFoundError(f"File not found: {test_ckpt}")
    return ckpt_path

class LDCTDeNormalize:
    def __init__(self,):
        self.norm_range_max = 3072.0
        self.norm_range_min = -1024.0
        self.trunc_min = -160.0
        self.trunc_max = 240.0
        
        self.data_range = self.trunc_max - self.trunc_min
        
    def denormalize(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image


    def trunc(self, mat):
        # Pytorch version
        if len(mat.shape) == 4 and isinstance(mat, torch.Tensor):
            mat = torch.clamp(mat[..., :, :], self.trunc_min, self.trunc_max)
        else:
            mat[mat <= self.trunc_min] = self.trunc_min
            mat[mat >= self.trunc_max] = self.trunc_max
        return mat
    
    def __call__(self, x):
        # return denormalized image 
        return self.trunc(self.denormalize(x))    

class Solver(object):
    def __init__(self, args, data_loader=None):
        self.args = args
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.data_loader = data_loader
        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu
        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.save_iters = args.save_iters
        self.result_fig = args.result_fig
        self.fig_suffix = args.fig_suffix
        self.save_cnt = 0 # save counter for saving figs
        self.DATE_IDENTIFIER = datetime.now().strftime('%Y%m%d%H%M%S')
        self.patch_size = args.patch_size
        self.test_ckpt = args.test_ckpt
        self.verbose = args.verbose
        
        self.val_metrics = dict() 
        self.model = get_ldct_model(self.args.test_model).to(self.device)
        self.lr = args.lr
        self.criterion = nn.MSELoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)

    def load_ckpt(self, pretrained_ckpt):
        if os.path.isfile(pretrained_ckpt):
            pretrained_dict = torch.load(pretrained_ckpt)
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            logging.info("Pretrained model loaded! path: {}".format(pretrained_ckpt))
        else:
            logging.debug("No pretrained model found!")

    def save_model(self, iter_):
        f = os.path.join(self.save_path, 'REDCNN_{}epoch.ckpt'.format(iter_))
        torch.save(self.model.state_dict(), f)

    def load_model(self):
        f = os.path.join(self.save_path, self.test_ckpt)
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f):
                n = k[7:]
                state_d[n] = v
            self.model.load_state_dict(state_d)
        else:
            self.model.load_state_dict(torch.load(f))

    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        x = x[:,:1,...]
        if isinstance(x, torch.Tensor):
            x = torch.squeeze(x).detach().cpu().numpy()
            y = torch.squeeze(y).detach().cpu().numpy()
            pred = torch.squeeze(pred).detach().cpu().numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join(self.save_path, f'fig_{self.fig_suffix}', 'result_{}.png'.format(fig_name)))
        plt.close()
        
    def save_res_np(self, x, y, pred, fig_name, original_result, pred_result, MAX_SAVE=100):
        save_path = Path(self.save_path) / self.DATE_IDENTIFIER
        save_path.mkdir(parents=True, exist_ok=True)
        name = f'x_y_pred_{fig_name}.npy'
        
        if self.save_cnt < MAX_SAVE:
            x = x[:,:1,...]
            if isinstance(x, torch.Tensor):
                x = torch.squeeze(x).detach().cpu().numpy()
                y = torch.squeeze(y).detach().cpu().numpy()
                pred = torch.squeeze(pred).detach().cpu().numpy()
            res_np = np.concatenate((x, y, pred), axis=1)
            np.save(save_path / name, res_np)
            self.save_cnt += 1
            
        original_result = convert_tensor_to_numpy(original_result)
        pred_result = convert_tensor_to_numpy(pred_result)
        with open(save_path / 'results_ori_pred.txt', 'a') as f:
            f.write(f'{fig_name}, {original_result[0]}, {pred_result[0]}, {original_result[1]}, {pred_result[1]}, {original_result[2]}, {pred_result[2]}\n')

    @torch.no_grad()
    def test(self):
        del self.model
        self.model = get_ldct_model(self.args.test_model).to(self.device)
        ckpt_path = get_ckpt_path(self.test_ckpt)
        print("Loading checkpoint from {}".format(ckpt_path))
        self.model.init_ckpt_from_path(ckpt_path)

        # compute PSNR, SSIM, RMSE
        ori_psnr_list, ori_ssim_list, ori_rmse_list = [], [], []
        pred_psnr_list, pred_ssim_list, pred_rmse_list = [], [], []
        print("Testing...")
        print(f"Used device: {self.device}")
        print(f"The number of test data is {len(self.data_loader)}")
        
        for i, (x, y) in enumerate(tqdm(self.data_loader)):
            x = x.float().to(self.device)
            y = y.unsqueeze(0).float().to(self.device)
            
            # TFA judge
            if self.args.test_model not in TFA_MODEL_NAMES:
                pred = self.model(x)
            else:
                pred = self.model(x, torch.load(f'/home/xuhang/{self.args.data_abbr}.pt').to(self.device))
            
            DeNorm = LDCTDeNormalize()
            data_range = DeNorm.data_range
            x = DeNorm(x)
            y = DeNorm(y)
            pred = DeNorm(pred) 

            if self.args.test_model not in FDD_MODEL_NAMES: 
                x = x.unsqueeze(0).float().to(self.device)
                pred = pred.unsqueeze(0).float().to(self.device)
                
            original_result, pred_result = compute_metrics(y, pred, x[:,:1,...], data_range=data_range, is_tensor=True)
            self.save_res_np(x, y, pred, i, original_result, pred_result)
            # save result figure
            if self.result_fig:
                if i>10:
                    ...
                else:
                    self.save_fig(x, y, pred, i, original_result, pred_result)

                
            ori_psnr = original_result[0].detach().cpu().numpy()
            ori_ssim = original_result[1].detach().cpu().numpy()
            ori_rmse = original_result[2].detach().cpu().numpy()
            pred_psnr = pred_result[0].detach().cpu().numpy()
            pred_ssim = pred_result[1].detach().cpu().numpy()
            pred_rmse = pred_result[2].detach().cpu().numpy()

            # avoid blank and black inf value
            ori_psnr_list.append(ori_psnr)
            ori_ssim_list.append(ori_ssim)
            ori_rmse_list.append(ori_rmse)
            pred_psnr_list.append(pred_psnr)
            pred_ssim_list.append(pred_ssim)
            pred_rmse_list.append(pred_rmse)
        
        # calculate the average by np.mean and avoid calcualting inf values
        ori_psnr_avg = np.nanmean(np.array(ori_psnr_list)[np.isfinite(ori_psnr_list)])
        ori_ssim_avg = np.nanmean(np.array(ori_ssim_list)[np.isfinite(ori_ssim_list)])
        ori_rmse_avg = np.nanmean(np.array(ori_rmse_list)[np.isfinite(ori_rmse_list)])
        pred_psnr_avg = np.nanmean(np.array(pred_psnr_list)[np.isfinite(pred_psnr_list)])
        pred_ssim_avg = np.nanmean(np.array(pred_ssim_list)[np.isfinite(pred_ssim_list)])
        pred_rmse_avg = np.nanmean(np.array(pred_rmse_list)[np.isfinite(pred_rmse_list)])
        print('\nOriginal === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f} \n'.format(ori_psnr_avg, ori_ssim_avg, ori_rmse_avg))
        print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}\n'.format(pred_psnr_avg, pred_ssim_avg, pred_rmse_avg))

def convert_tensor_to_numpy(tens):
    if isinstance(tens, torch.Tensor):
        return tens.detach().cpu().numpy()
    else:
        return tens
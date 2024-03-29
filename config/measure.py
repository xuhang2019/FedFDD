import torch
import numpy as np
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import mean_squared_error as mse 
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

def compute_metrics(y, pred, x = None, data_range = 400.0, ssim_mode = 'default', is_tensor = False):
    """
        compute PSNR, SSIM, RMSE
        x is optional, if x is None, then only compute metrics for y and pred.
        Typically, x is available to acquire since pred = net(x).
        
        Tensor mode: B*C*H*W, recommendatory: C = 1
        Numpy mode: only support 2dim image
    """
    if is_tensor:
        # assert y and pred is tensor type
        assert isinstance(y, torch.Tensor) and isinstance(pred, torch.Tensor), "y and pred must be tensor type"
        
        if x is not None:
            # xy means original input and ground truth
            xy_psnr = psnr(x, y, data_range=data_range)
            xy_ssim = ssim(x, y, data_range=data_range)
            xy_rmse = mse(x, y, squared=False)
            
        yp_psnr = psnr(y, pred, data_range=data_range)
        yp_ssim = ssim(y, pred, data_range=data_range)
        yp_rmse = mse(y, pred, squared=False)
        
    else: 
        assert len(pred.shape) == 2, "numpy mode only suppors one 2-d image at a time"
        if x is not None:
            # xy means original input and ground truth
            xy_psnr = peak_signal_noise_ratio(x, y, data_range=data_range)
            xy_ssim = structural_similarity(x, y, data_range=data_range) if ssim_mode=='default' else structural_similarity(x, y, data_range=data_range, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
            xy_rmse = np.sqrt(mean_squared_error(x, y))
        
        # yp means ground truth and prediction
        yp_psnr = peak_signal_noise_ratio(y, pred, data_range=data_range)
        yp_ssim = structural_similarity(y, pred, data_range=data_range) if ssim_mode=='default' else structural_similarity(y, pred, data_range=data_range, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        yp_rmse = np.sqrt(mean_squared_error(y, pred))
    
    if x is not None:
        return (xy_psnr, xy_ssim, xy_rmse), (yp_psnr, yp_ssim, yp_rmse)
    else:
        return (yp_psnr, yp_ssim, yp_rmse)



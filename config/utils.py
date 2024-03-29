#%%
import os
import random
from copy import deepcopy
from collections import Counter, OrderedDict
from tqdm import tqdm
from typing import Any, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime

import torch
import pynvml
import numpy as np
from torch.utils.data import DataLoader
from rich.console import Console
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import wandb

# from data.utils.datasets import BaseDataset
# breakpoint()
from measure import compute_metrics

from ldct_models import FDD_MODEL_NAMES, TFA_MODEL_NAMES

PROJECT_DIR = Path(__file__).absolute().parent.parent.parent
OUT_DIR = PROJECT_DIR / "out"
TEMP_DIR = PROJECT_DIR / "temp"


def fix_random_seed(seed: int) -> None:
    """Fix the random seed of FL training.

    Args:
        seed (int): Any number you like as the random seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_best_device(use_cuda: bool) -> torch.device:
    """Dynamically select the vacant CUDA device for running FL experiment.

    Args:
        use_cuda (bool): `True` for using CUDA; `False` for using CPU only.

    Returns:
        torch.device: The selected CUDA device.
    """
    # This function is modified by the `get_best_gpu()` in https://github.com/SMILELab-FL/FedLab/blob/master/fedlab/utils/functional.py
    # Shout out to FedLab, which is an incredible FL framework!
    # FIXME: fix the cuda - bug
    # return torch.device("cuda:2")
    if not torch.cuda.is_available() or not use_cuda:
        return torch.device("cpu")
    pynvml.nvmlInit()
    gpu_memory = []
    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        gpu_ids = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
        # breakpoint()
        assert max(gpu_ids) < torch.cuda.device_count()
    else:
        gpu_ids = range(torch.cuda.device_count())

    for i in gpu_ids:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory.append(memory_info.free)
    gpu_memory = np.array(gpu_memory)
    best_gpu_id = np.argmax(gpu_memory)
    return torch.device(f"cuda:{best_gpu_id}")
    


def trainable_params(
    src,
    detach=False,
    requires_name=False,
):
    """Collect all parameters in `src` that `.requires_grad = True` into a list and return it.

    Args:
        src (Union[OrderedDict[str, torch.Tensor], torch.nn.Module]): The source that contains parameters.
        requires_name (bool, optional): If set to `True`, The names of parameters would also return in another list. Defaults to False.
        detach (bool, optional): If set to `True`, the list would contain `param.detach().clone()` rather than `param`. Defaults to False.

    Returns:
        Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[str]]]: List of parameters, [List of names of parameters].
    """
    func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
    parameters = []
    keys = []
    if isinstance(src, OrderedDict):
        for name, param in src.items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)
    elif isinstance(src, torch.nn.Module):
        for name, param in src.state_dict(keep_vars=True).items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)

    if requires_name:
        return parameters, keys
    else:
        return parameters


def vectorize(
    src, detach=True
) -> torch.Tensor:
    """Vectorize and concatenate all tensors in `src`.

    Args:
        src (Union[OrderedDict[str, torch.Tensor]List[torch.Tensor]]): The source of tensors.
        detach (bool, optional): Set to `True`, return the `.detach().clone()`. Defaults to True.

    Returns:
        torch.Tensor: The vectorized tensor.
    """
    func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
    if isinstance(src, list):
        return torch.cat([func(param).flatten() for param in src])
    elif isinstance(src, OrderedDict):
        return torch.cat([func(param).flatten() for param in src.values()])


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion=torch.nn.MSELoss(),
    device=torch.device("cpu"),
    save_fig=False,
    model_name=None,
):
    """For evaluating the `model` over `dataloader` and return the result calculated by `criterion`.

    Args:
        model (torch.nn.Module): Target model.
        dataloader (DataLoader): Target dataloader.
        criterion (optional): The metric criterion. Defaults to torch.nn.CrossEntropyLoss(reduction="sum").
        device (torch.device, optional): The device that holds the computation. Defaults to torch.device("cpu").

    Returns:
        Tuple[float, float, int]: [metric, correct, sample num]
    """
    model = model.to(device)
    model.eval()
    sample_num = len(dataloader)
    ori_psnr, ori_ssim, ori_rmse = 0.0, 0.0, 0.0
    pred_psnr, pred_ssim, pred_rmse = 0.0, 0.0, 0.0
    
    time_suffix = datetime.now().strftime("%m%d%H%M%S")

    for idx, (x, y) in enumerate(tqdm(dataloader)):
        if model.model_choice in FDD_MODEL_NAMES:
            x = x.float().to(device)
            y = y.unsqueeze(0).float().to(device)
        else:
            x = x.unsqueeze(0).float().to(device)
            y = y.unsqueeze(0).float().to(device)
        
        if model.model_choice in TFA_MODEL_NAMES:
            if dataloader.dataset.patients[0][0] == 'C':
                text_features = torch.load('/home/xuhang/chest.pt').to(device)
            elif dataloader.dataset.patients[0][0] == 'L':
                text_features = torch.load('/home/xuhang/abdomen.pt').to(device)
            elif dataloader.dataset.patients[0][0] == 'N':
                text_features = torch.load('/home/xuhang/head.pt').to(device)
            else:
                raise ValueError(f"Unknown patient type: {dataloader.dataset.patients[0][0]}")
            pred = model(x, text_features)
        else:
            pred = model(x) 
            
        DeNorm = LDCTDeNormalize()
        data_range = DeNorm.data_range
        x = DeNorm(x)
        y = DeNorm(y)
        pred = DeNorm(pred) 
        
        if idx == 4:
            wandb.Image(pred.cpu().numpy(), mode='L', caption=f"Predicted {dataloader.dataset.patients[0]}")
            wandb.Image(y.cpu().numpy(), mode='L', caption=f"Full-dose {dataloader.dataset.patients[0]}")
        

        original_result, pred_result = compute_metrics(y, pred, x[:,:1,...], data_range=data_range, is_tensor=True)
        
        ori_psnr += original_result[0].detach().cpu().numpy()
        ori_ssim += original_result[1].detach().cpu().numpy()
        ori_rmse += original_result[2].detach().cpu().numpy()
        pred_psnr += pred_result[0].detach().cpu().numpy()
        pred_ssim += pred_result[1].detach().cpu().numpy()
        pred_rmse += pred_result[2].detach().cpu().numpy()
        
        if save_fig:
            save_ldct_fig(x, y, pred, idx, original_result, pred_result, model_name, time_suffix)
        
    # give a eval_dict
    eval_dict = {
        'ori_psnr': ori_psnr / sample_num,
        'ori_ssim': ori_ssim / sample_num,
        'ori_rmse': ori_rmse / sample_num,
        'pred_psnr': pred_psnr / sample_num,
        'pred_ssim': pred_ssim / sample_num,
        'pred_rmse': pred_rmse / sample_num,
    }
        
    print("\n".join("{}:\t{:.4f}".format(k, v) for k, v in eval_dict.items()))
    
    return eval_dict

def save_ldct_fig(x, y, pred, fig_name, original_result, pred_result, model_name, time_suffix):
    """
        Args:
            :model_name: args.model for saving path: out/model_name/fig_time_suffix
            :time_suffix: time suffix, passing by evaluate function (datetime.now().strftime("%m%d%H%M%S"))
    """
    trunc_min, trunc_max = -160.0, 240.0
    if isinstance(x, torch.Tensor):
        x = torch.squeeze(x).detach().cpu().numpy()
        y = torch.squeeze(y).detach().cpu().numpy()
        pred = torch.squeeze(pred).detach().cpu().numpy()
    f, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].imshow(x, cmap=plt.cm.gray, vmin=trunc_min, vmax=trunc_max)
    ax[0].set_title('Quarter-dose', fontsize=30)
    ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                        original_result[1],
                                                                        original_result[2]), fontsize=20)
    ax[1].imshow(pred, cmap=plt.cm.gray, vmin=trunc_min, vmax=trunc_max)
    ax[1].set_title('Result', fontsize=30)
    ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                        pred_result[1],
                                                                        pred_result[2]), fontsize=20)
    ax[2].imshow(y, cmap=plt.cm.gray, vmin=trunc_min, vmax=trunc_max)
    ax[2].set_title('Full-dose', fontsize=30)
    
    fig_path = OUT_DIR/ model_name/ f'fig_{time_suffix}'/f"result_{fig_name}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    f.savefig(fig_path)
    plt.close()


# def count_labels(
#     dataset: BaseDataset, indices: List[int] = None, min_value=0
# ) -> List[int]:
#     """For counting number of labels in `dataset.targets`.

#     Args:
#         dataset (BaseDataset): Target dataset.
#         indices (List[int]): the subset indices. Defaults to all indices of `dataset` if not specified.
#         min_value (int, optional): The minimum value for each label. Defaults to 0.

#     Returns:
#         List[int]: The number of each label.
#     """
#     if indices is None:
#         indices = list(range(len(dataset.targets)))
#     counter = Counter(dataset.targets[indices].tolist())
#     return [counter.get(i, min_value) for i in range(len(dataset.classes))]


class Logger:
    def __init__(
        self, stdout: Console, enable_log: bool, logfile_path: Union[Path, str]
    ):
        """This class is for solving the incompatibility between the progress bar and log function in library `rich`.

        Args:
            stdout (Console): The `rich.console.Console` for printing info onto stdout.
            enable_log (bool): Flag indicates whether log function is actived.
            logfile_path (Union[Path, str]): The path of log file.
        """
        self.stdout = stdout
        self.logfile_stream = None
        self.enable_log = enable_log
        if self.enable_log:
            self.logfile_stream = open(logfile_path, "w")
            self.logger = Console(
                file=self.logfile_stream, record=True, log_path=False, log_time=False
            )

    def log(self, *args, **kwargs):
        self.stdout.log(*args, **kwargs)
        if self.enable_log:
            self.logger.log(*args, **kwargs)

    def close(self):
        if self.logfile_stream:
            self.logfile_stream.close()


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
    
    
def afm():
    h = 512
    w = 512
    
    radius = np.sqrt(w * w + h * h)
    X, Y = np.meshgrid(np.linspace(0, h - 1, h), np.linspace(0, w - 1, w))
    D = np.sqrt(X * X + Y * Y)
    
    lrdown = np.random.uniform(0.43, 0.50)
    lrdown = lrdown * radius
    mask1 = np.ones((w, h))
    mask1[D > lrdown] = 0
    
    D = D/np.max(D)
    
    adaptive_mask =np.random.binomial(1, D)
    
    # | is only valid for bool or bitwise operations
    mask = np.where((mask1 == 1) | (adaptive_mask == 1), 1, 0) 
    return mask

def dct2(a):
    """Compute 2D Discrete Cosine Transform."""
    return dct(dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2(a):
    """Compute 2D Inverse Discrete Cosine Transform."""
    return idct(idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


#%%
if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(__file__))
    import sys
    sys.path.append('../../')
    from ldct_models import REDCNNFreq, REDCNNFreqTFA
    from ldct_loader import get_fl_loader

    
    # ckpt = '/home/xuhang/FL_benchmark/out/pfedfreq_direct_2/RED_CNN_pfedfreq_direct_2_val_best.ckpt'
    # model = REDCNNFreq(alpha=0)
    model = REDCNNFreqTFA(alpha=0)
    # model.load_state_dict(torch.load(ckpt))
    
    class MockArgs:
        mode = 'validation'
        yaml_path = '/home/xuhang/FL_benchmark/src/config/baseline.yaml'
        data_abbr = 'chest'
        num_workers = 4
        random_seed = 626
        in_federation = False
        load_mode = 0
        data_path = '/home/xuhang/data_ct_denoise'
        patch_n = 16
        patch_size = 64
        debug = False
        model_choice = 'freq-tfa'
        model_alpha = 0
    
    args = MockArgs()
    val_loader = get_fl_loader(args, is_validation=True)
    evaluate(model, val_loader, device=torch.device('cuda:2'))
    

    
# %%

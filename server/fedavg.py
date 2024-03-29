import sys
import os
import random
from pathlib import Path
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List

import torch
import wandb

from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from rich.progress import track

PROJECT_DIR = Path(__file__).absolute().parent.parent
sys.path.append(PROJECT_DIR.as_posix())
sys.path.append(PROJECT_DIR.joinpath("config").as_posix())

from config.utils import (
    OUT_DIR,
    Logger,
    fix_random_seed,
    trainable_params,
    get_best_device,
)
from client.fedavg import FedAvgClient
from config.ldct_models import get_ldct_model

def get_fedavg_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-jr", "--join_ratio", type=float, default=0.1)
    parser.add_argument("-ge", "--global_epoch", type=int, default=500)
    parser.add_argument("-fe", "--finetune_epoch", type=int, default=0)
    parser.add_argument("-tg", "--test_gap", type=int, default=100)
    parser.add_argument("-ee", "--eval_test", type=int, default=1)
    parser.add_argument("-er", "--eval_train", type=int, default=0)
    parser.add_argument("-mom", "--momentum", type=float, default=0.0)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-vg", "--verbose_gap", type=int, default=100000)
    parser.add_argument("-v", "--visible", type=int, default=0)
    parser.add_argument("--global_testset", type=int, default=0)
    parser.add_argument("--straggler_ratio", type=float, default=0)
    parser.add_argument("--straggler_min_local_epoch", type=int, default=1)
    parser.add_argument("--use_cuda", type=int, default=1)
    parser.add_argument("--save_log", type=int, default=1)
    parser.add_argument("--save_model", type=int, default=0)
    parser.add_argument("--save_metrics", type=int, default=1)
    parser.add_argument('--algo', type=str,default='', help='algorithm name')
    
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--load_mode', type=int, default=0)
    parser.add_argument('--exp_code', type=str, default='061801')
    parser.add_argument('--test_code', type=str, default='1')
    
    # load ckpt
    parser.add_argument('--pretrained_model', type=str, default='')
    parser.add_argument('--test_model', type=str, default='')

    parser.add_argument('--data_path', type=str, default='/home/xuhang/data_ct_denoise')
    parser.add_argument('--data_abbr', type=str, default='fed')
    parser.add_argument('--yaml_path', type=str, default='config/pts.yaml') # use with args.exp_code
    
    parser.add_argument('--save_path', type=str, default='./save/') # if set exp_code, this will be changed.
    parser.add_argument('--fig_suffix', type=str, default='')


    # normalization parameters
    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)
    parser.add_argument('--trunc_min', type=float, default=-160.0)
    parser.add_argument('--trunc_max', type=float, default=240.0)
    parser.add_argument('--transform', type=bool, default=False)
    
    # this setting has been changed compared with the original one
    parser.add_argument('--patch_n', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=16)
    
    # Federated learning parameters
    parser.add_argument('--fl', action='store_true')
    parser.add_argument('--local_epoch', type=int, default=1, help="the number of local epochs")
    parser.add_argument('--local_batch_size', type=int, default=16, help="local batch size")

    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--print_iters', type=int, default=20)
    parser.add_argument('--decay_iters', type=int, default=3000)
    parser.add_argument('--save_epochs', type=int, default=10)
    parser.add_argument('--test_iters', type=int, default=1000)
    
    # add seed
    parser.add_argument('--random_seed', type=int, default=626)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--local_lr', type=float, default=1e-5)
    parser.add_argument('--device', type=str)
    parser.add_argument('--num_workers', type=int, default=7)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    
    parser.add_argument('--save_fig', action='store_true', help='save result figure')
    parser.add_argument('--verbose', action='store_true', help='verbose print')

    parser.add_argument('--in_federation', action='store_false', help='split the train and test data in federation')

    
    parser.add_argument('--debug', action='store_true', help='debug mode: a dataset terminates in 10 samples')
    parser.add_argument('--no_wandb', action='store_true', help='debug mode: disable wandb')
    
    parser.add_argument('--allow_p', action='store_true', help='test mode or allow pretrained')
    parser.add_argument('--unique_model', action='store_true', help='unique model for each client')
    
    parser.add_argument('--model_choice', type=str, default='ori')
    parser.add_argument('--model_alpha', type=int, default=0)
    return parser


class FedAvgServer:
    def __init__(
        self,
        algo: str = '',
        args: Namespace = None,
        unique_model=False,
        default_trainer=True,
    ):
        self.args = get_fedavg_argparser().parse_args() if args is None else args
        self.algo = self.args.algo
        self.unique_model = self.args.unique_model
        
        if not self.args.no_wandb:
            wandb.init(project='LDCT_Fed',id = self.algo, sync_tensorboard=True)
        fix_random_seed(self.args.seed)
        self.train_clients: List[int] = [0,1,2] # Only three clients
        self.test_clients: List[int] = [0,1,2]
        self.client_num = 3

        self.device = get_best_device(self.args.use_cuda)
        self.default_trainer = default_trainer
        print(f'Using device: {self.device}')
        use_model = get_ldct_model(self.args.model_choice, self.args.model_alpha)
        self.model = use_model.to(self.device)
    
        if self.args.allow_p or self.args.mode == 'test':
            assert Path(self.args.pretrained_model).is_file(), "Pretrained model (`args.pretrained_model`) not found."
            init_trainable_params, self.trainable_params_name = trainable_params(
                torch.load(Path(self.args.pretrained_model)), detach=True, requires_name=True
            )
        else:
            init_trainable_params, self.trainable_params_name = trainable_params(
                self.model, detach=True, requires_name=True
            )
        # client_trainable_params is for pFL, which outputs exclusive model per client
        # global_params_dict is for regular FL, which outputs a single global model
        if self.unique_model:
            self.client_trainable_params = [
                deepcopy(init_trainable_params) for _ in self.train_clients
            ]
        self.global_params_dict = OrderedDict(
            zip(self.trainable_params_name, init_trainable_params)
        )

        self.clients_local_epoch = [self.args.local_epoch] * self.client_num
        if (
            self.args.straggler_ratio > 0
            and self.args.local_epoch > self.args.straggler_min_local_epoch
        ):
            straggler_num = int(self.client_num * self.args.straggler_ratio)
            normal_num = self.client_num - straggler_num
            self.clients_local_epoch = [self.args.local_epoch] * (
                normal_num
            ) + random.choices(
                range(self.args.straggler_min_local_epoch, self.args.local_epoch),
                k=straggler_num,
            )
            random.shuffle(self.clients_local_epoch)
     
        self.client_sample_stream = [[0,1,2] for _ in range(self.args.global_epoch+1)]
        self.selected_clients = []
        self.current_epoch = 0
        # For controlling behaviors of some specific methods while testing (not used by all methods)
        self.test_flag = False

        # variables for logging
        if not os.path.isdir(OUT_DIR / self.algo) and (
            self.args.save_log or self.args.save_fig or self.args.save_metrics
        ):
            os.makedirs(OUT_DIR / self.algo, exist_ok=True)
        self.writer = SummaryWriter(log_dir=OUT_DIR / self.algo/ 'runs')

        self.client_stats = {i: {} for i in self.train_clients}
        self.metrics = {
            "train_before": [],
            "train_after": [],
            "test_before": [],
            "test_after": [],
        }
        stdout = Console(log_path=False, log_time=False)
        self.logger = Logger(
            stdout=stdout,
            enable_log=self.args.save_log,
            logfile_path=OUT_DIR / self.algo / f"{self.args.dataset}_log.html",
        )
        self.test_results = {}
        self.train_progress_bar = track(
            range(self.args.global_epoch), "[bold green]Training...", console=stdout
        )

        self.logger.log("=" * 20, "ALGORITHM:", self.algo, "=" * 20)
        self.logger.log("Experiment Arguments:", dict(self.args._get_kwargs()))


        self.trainer = None
        if self.default_trainer:
            self.trainer = FedAvgClient(deepcopy(self.model), self.args, self.logger, writer=self.writer, device=self.device)

    def train(self):
        if self.args.debug: # only for debug
            self.args.global_epoch = 11
        
        self.best_psnr = {
            'value': float('-inf'),
            'epoch': 0,
        }      
        for E in range(1, self.args.global_epoch+1):
            self.current_epoch = E
            if (E) % self.args.verbose_gap == 0:
                self.logger.log("-" * 26, f"TRAINING EPOCH: {E}", "-" * 26)

            self.selected_clients = self.client_sample_stream[E-1] # 0 - 100
            print("="*10,f"Epoch:{E}/{self.args.global_epoch}", "="*10)
            self.train_one_round()
            
            
            self.val()
            
            if ((E) % self.args.save_epochs == 0) or (E == self.args.global_epoch):
                self.save_checkpoint(local_epoch=E)
                
    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at server side) in each communication round."""
        delta_cache = []
        weight_cache = []
        for client_id in self.selected_clients:
            client_local_params = self.generate_client_params(client_id)
            # self.trainer is the client
            (
                delta,
                weight,
                self.client_stats[client_id][self.current_epoch],
            ) = self.trainer.train( # each client use the same trainer
                client_id=client_id,
                local_epoch=self.clients_local_epoch[client_id],
                new_parameters=client_local_params,
                verbose=((self.current_epoch + 1) % self.args.verbose_gap) == 0,
            )
            
            delta_cache.append(delta)
            weight_cache.append(weight)
        self.writer.add_scalars(f'train/client_loss', {str(client_id): self.client_stats[client_id][self.current_epoch] for client_id in self.selected_clients}, self.current_epoch)
        
        self.aggregate(delta_cache, weight_cache)

    def test(self, is_validation=False):
        """The function for testing FL method's output (a single global model or personalized client models)."""
        self.test_flag = True
        stat_list = []
        for client_id in self.test_clients: # set test_clients = 1
            client_local_params = self.generate_client_params(client_id)
            stat = self.trainer.test(client_id, client_local_params, is_validation)
            
            self.writer.add_scalars('val/psnr', {f'ori{client_id}':stat['ori_psnr'],
                                                 f'val{client_id}':stat['pred_psnr']})
            self.writer.add_scalars('val/ssim', {f'ori{client_id}':stat['ori_ssim'],
                                                 f'val{client_id}':stat['pred_ssim']})
            self.writer.add_scalars('val/rmse', {f'ori{client_id}':stat['ori_rmse'],
                                                f'val{client_id}':stat['pred_rmse']},)
            
            stat_list.append(stat)

        self.test_results[self.current_epoch + 1] = stat_list
        self.test_flag = False
        return stat_list

    def val(self):
        return self.test(is_validation=True)
    
    @torch.no_grad()
    def update_client_params(self, client_params_cache: List[List[torch.Tensor]]):
        """
        The function for updating clients model while unique_model is `True`.
        This function is only useful for some pFL methods.

        Args:
            client_params_cache (List[List[torch.Tensor]]): models parameters of selected clients.

        Raises:
            RuntimeError: If unique_model = `False`, this function will not work properly.
        """
        if self.unique_model:
            for i, client_id in enumerate(self.selected_clients):
                self.client_trainable_params[client_id] = client_params_cache[i]
        else:
            raise RuntimeError(
                "FL system don't preserve params for each client (unique_model = False)."
            )

    def generate_client_params(self, client_id: int):
        if self.unique_model:
            return OrderedDict(
                zip(self.trainable_params_name, self.client_trainable_params[client_id])
            )
        else:
            return self.global_params_dict

    @torch.no_grad()
    def aggregate(
        self,
        delta_cache: List[List[torch.Tensor]],
        weight_cache: List[int],
        return_diff=True,
    ):
        """
        This function is for aggregating recevied model parameters from selected clients.
        The method of aggregation is weighted averaging by default.

        Args:
            delta_cache (List[List[torch.Tensor]]): `delta` means the difference between client model parameters that before and after local training.

            weight_cache (List[int]): Weight for each `delta` (client dataset size by default).

            return_diff (bool): Differnt value brings different operations. Default to True.
        """
        weights = torch.tensor(weight_cache, device=self.device) / sum(weight_cache)
        if return_diff:
            delta_list = [list(delta.values()) for delta in delta_cache]
            aggregated_delta = [
                torch.sum(weights * torch.stack(diff, dim=-1), dim=-1)
                for diff in zip(*delta_list)
            ]

            for param, diff in zip(self.global_params_dict.values(), aggregated_delta):
                param.data -= diff
        else:
            for old_param, zipped_new_param in zip(
                self.global_params_dict.values(), zip(*delta_cache)
            ):
                old_param.data = (torch.stack(zipped_new_param, dim=-1) * weights).sum(
                    dim=-1
                )
        self.model.load_state_dict(self.global_params_dict, strict=False)
        # update global_params_dict

    def save_checkpoint(self, local_epoch=None, is_best=False):
        if local_epoch is None:
            local_epoch = self.args.global_epoch
        
        if is_best:
            model_name = f"RED_CNN_{self.algo}_val_best.ckpt"
        else:
            model_name = f"RED_CNN_{self.algo}_{local_epoch}_{self.args.model}.ckpt"

        if self.unique_model:
            torch.save(
                self.client_trainable_params, OUT_DIR / self.algo / model_name
            )
        else:
            torch.save(self.model.state_dict(), OUT_DIR / self.algo / model_name)
        
        print(f'Model saved: {model_name}')
        
    def run(self):
        """The comprehensive FL process.

        Raises:
            RuntimeError: If `trainer` is not set.
        """
        if self.args.mode == 'train':
            if self.trainer is None:
                raise RuntimeError(
                    "Specify your unique trainer or set `default_trainer` as True."
                )
            self.trainer.device = self.device
            self.train() # will save model each 5 epochs

            self.logger.log(
                "=" * 20, self.algo, "TEST RESULTS:", "=" * 20, self.test_results
            )
            self.logger.close()
        elif self.args.mode == 'test':
            self.test()
                
if __name__ == "__main__":
    server = FedAvgServer()
    server.run()

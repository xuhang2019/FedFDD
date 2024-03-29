import pickle
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Tuple, Union
from pathlib import Path
from tqdm import tqdm

import datetime

import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).parent.parent.absolute()

from config.utils import trainable_params, evaluate, Logger
from config.ldct_loader import get_fl_loader
from config.ldct_models import FDD_MODEL_NAMES

global TEST_NUM_0
global TEST_NUM_1
global TEST_NUM_2
TEST_NUM_0, TEST_NUM_1, TEST_NUM_2 = 1,1,1


class FedAvgClient:
    def __init__(self, model: torch.nn.Module, args: Namespace, logger: Logger, writer=None, device=None):
        self.args = args
        self.device = device # default: from server
        self.client_id: int = None
        self.writer = writer
        self.model = model.to(self.device)
        self.local_epoch = self.args.local_epoch
        
        self.logger = logger
        self.personal_params_dict: Dict[int, Dict[str, torch.Tensor]] = {}
        self.personal_params_name: List[str] = []
        self.init_personal_params_dict: Dict[str, torch.Tensor] = {
            key: param.clone().detach()
            for key, param in self.model.state_dict(keep_vars=True).items()
            if not param.requires_grad
        }
        
        self.opt_state_dict = {}

        # self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.criterion = torch.nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.Adam(
            params=trainable_params(self.model),
            lr=self.args.local_lr,
            # momentum=self.args.momentum,
            # weight_decay=self.args.weight_decay,
        )
        self.init_opt_state_dict = deepcopy(self.optimizer.state_dict())
        
        # load fl_loader
        self.loaders = get_fl_loader(self.args)
    
    def load_dataset(self):
        # not distinguish train and test
        if isinstance(self.loaders, list):
            self.loader = self.loaders[self.client_id]
        else:
            assert isinstance(self.loaders, DataLoader)
            self.loader = self.loaders
            
    def get_validation_loader(self, client_id):
        self.validation_loader = get_fl_loader(self.args, is_validation=True, client_id=client_id)

    def train_and_log(self, verbose=False):
        if self.local_epoch > 0:
            loss = self.fit() # here will run local epoch rounds
            self.save_state() # before aggregation
        return loss
            
    def set_parameters(self, new_parameters):
        """Load model parameters received from the server.

        Args:
            new_parameters (OrderedDict[str, torch.Tensor]): Parameters of FL model.
        """
        personal_parameters = self.personal_params_dict.get(
            self.client_id, self.init_personal_params_dict
        )
        self.optimizer.load_state_dict(
            self.opt_state_dict.get(self.client_id, self.init_opt_state_dict)
        )
        self.model.load_state_dict(new_parameters, strict=False)
        self.model.load_state_dict(personal_parameters, strict=False)

    def save_state(self):
        """Save client model personal parameters and the state of optimizer at the end of local training."""
        self.personal_params_dict[self.client_id] = {
            key: param.clone().detach()
            for key, param in self.model.state_dict(keep_vars=True).items()
            if (not param.requires_grad) or (key in self.personal_params_name)
        }
        self.opt_state_dict[self.client_id] = deepcopy(self.optimizer.state_dict())

    def train(
        self,
        client_id: int,
        local_epoch: int,
        new_parameters,
        return_diff=True,
        verbose=False,
    ):
        """
        The funtion for including all operations in client local training phase.
        If you wanna implement your method, consider to override this funciton.

        Args:
            client_id (int): The ID of client.

            local_epoch (int): The number of epochs for performing local training.

            new_parameters (OrderedDict[str, torch.Tensor]): Parameters of FL model.

            return_diff (bool, optional):
            Set as `True` to send the difference between FL model parameters that before and after training;
            Set as `False` to send FL model parameters without any change.  Defaults to True.

            verbose (bool, optional): Set to `True` for print logging info onto the stdout (Controled by the server by default). Defaults to False.

        Returns:
            Tuple[Union[OrderedDict[str, torch.Tensor], List[torch.Tensor]], int, Dict]:
            [The difference / all trainable parameters, the weight of this client, the evaluation metric stats].
        """
        self.client_id = client_id
        self.local_epoch = local_epoch
        self.load_dataset()
        self.set_parameters(new_parameters)
        loss = self.train_and_log(verbose=verbose) # modifed: only save loss

        if return_diff:
            delta = OrderedDict()
            for (name, p0), p1 in zip(
                new_parameters.items(), trainable_params(self.model)
            ):
                delta[name] = p0 - p1
            return delta, len(self.loader), loss
        else:
            return (
                trainable_params(self.model, detach=True),
                len(self.loader),
                loss,
            )

    def fit(self):
        """
        The function for specifying operations in local training phase.
        If you wanna implement your method and your method has different local training operations to FedAvg, this method has to be overrided.
        
        return loss in this local epoch
        """
        self.model.train()
        local_epoch_loss = []
        for _ in range(self.local_epoch):
            epoch_loss = []
            for x, y in tqdm(self.loader):
                # When the current batch size is 1, the batchNorm2d modules in the model would raise error.
                # So the latent size 1 data batches are discarded.
                if len(x) <= 1:
                    continue
                
                if self.args.model_choice in FDD_MODEL_NAMES:
                    x = x.float().to(self.device)
                    y = y.unsqueeze(0).float().to(self.device) 
                    if self.args.patch_size:
                        patch_size = self.args.patch_size
                        x = x.view(-1, 3, patch_size, patch_size)
                        y = y.view(-1, 1, patch_size, patch_size)
                else:
                    x = x.unsqueeze(0).float().to(self.device)
                    y = y.unsqueeze(0).float().to(self.device)
                    if self.args.patch_size:
                        patch_size = self.args.patch_size
                        x = x.view(-1, 1, patch_size, patch_size)
                        y = y.view(-1, 1, patch_size, patch_size)
 
                pred = self.model(x)
                loss = self.criterion(pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss.append(loss.detach().item())
            local_epoch_loss.append(sum(epoch_loss) / len(epoch_loss))
        fit_loss = sum(local_epoch_loss) / len(local_epoch_loss)
        
        print(f'Client id: {self.client_id}, loss:{fit_loss:.6f}\n')
        return fit_loss
        
    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module = None, is_validation=False) -> Dict[str, float]:
        eval_model = self.model if model is None else model
        eval_model.eval()
        criterion = torch.nn.MSELoss()
        eval_dict = evaluate(
            model=eval_model,
            dataloader=self.loader if not is_validation else self.validation_loader,
            criterion=criterion,
            device=self.device,
            save_fig=self.args.save_fig if not is_validation else False,
            model_name=self.args.algo,
        )
        return eval_dict

    def test(
        self, client_id: int, new_parameters, is_validation = False,
    ):
        """Test function. Only be activated while in FL test round.

        Args:
            client_id (int): The ID of client.
            new_parameters (OrderedDict[str, torch.Tensor]): The FL model parameters.

        Returns:
            eval_dict: ori_psnr, ori_ssim, ori_rmse, pred ...
        """
        self.client_id = client_id
        if not is_validation:
            self.load_dataset() 
        else:
            self.get_validation_loader(client_id)
        self.set_parameters(new_parameters) # after aggregation
        eval_dict = self.evaluate(is_validation=is_validation)
        
        datestr = datetime.datetime.now().strftime("%H%M%S")
        
        if client_id == 0:
            global TEST_NUM_0
            if TEST_NUM_0 % 10 == 0:
                torch.save(self.model.state_dict(), f'/home/xuhang/debug_ckpts/c{client_id}_{self.model.model_choice}_{TEST_NUM_0}_{datestr}.ckpt')
            TEST_NUM_0 += 1
        elif client_id == 1:
            global TEST_NUM_1
            if TEST_NUM_1 % 10 == 0:
                torch.save(self.model.state_dict(), f'/home/xuhang/debug_ckpts/c{client_id}_{self.model.model_choice}_{TEST_NUM_1}_{datestr}.ckpt')
            TEST_NUM_1 += 1 
        elif client_id == 2:
            global TEST_NUM_2
            if TEST_NUM_2 % 10 == 0:
                torch.save(self.model.state_dict(), f'/home/xuhang/debug_ckpts/c{client_id}_{self.model.model_choice}_{TEST_NUM_2}_{datestr}.ckpt')
            TEST_NUM_2 += 1
        
        # default zero
        if self.args.finetune_epoch > 0:
            self.finetune()
            after = self.evaluate()
        return eval_dict

    def finetune(self):
        """
        The fine-tune function. If your method has different fine-tuning opeation, consider to override this.
        This function will only be activated while in FL test round.
        """
        self.model.train()
        for _ in range(self.args.finetune_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

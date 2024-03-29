import os
from glob import glob
import random
import numpy as np
import yaml
from torch.utils.data import Dataset, DataLoader
from config.utils import afm, dct2, idct2
from ldct_models import FDD_MODEL_NAMES


class LDCTDataset(Dataset):
    """
        The fed patients' dataset will be sorted.
        return data: (input, target)
    """
    def __init__(self, args, patients, transform=None, random_seed=None, in_federation=None, is_validation=False):
        self.args = args
        self.mode = args.mode        
        self.load_mode = args.load_mode
        assert self.load_mode in [0,1], "--load_mode is 0 or 1"
        self.patch_n = args.patch_n
        self.patch_size = args.patch_size
        self.transform = transform
        self.data_path = args.data_path
        self.is_validation = is_validation
        self.patients = patients
        self.get_input_target(patients, random_seed, in_federation if not is_validation else False)
                
    def get_input_target(self, patients, random_seed, in_federation):
        input_path = []
        target_path = []
        
        for patient in patients:
            curr_input_path = np.array(sorted(glob(os.path.join(self.data_path, f'{patient}*_input.npy'))))
            curr_target_path = np.array(sorted(glob(os.path.join(self.data_path, f'{patient}*_target.npy'))))
            assert len(curr_input_path) == len(curr_target_path), "input and target should have the same length"
             
            if in_federation: 
                num_curr = len(curr_input_path)
                num_train = int(0.7*num_curr)
                picked_idx = list(range(num_curr))
                
                random.seed(random_seed) 
                random.shuffle(picked_idx)
                
                if self.mode == 'train':
                    picked_input_path = curr_input_path[picked_idx[:num_train]]
                    picked_target_path = curr_target_path[picked_idx[:num_train]]
                elif self.mode == 'test':
                    picked_input_path = curr_input_path[picked_idx[num_train:]]
                    picked_target_path = curr_target_path[picked_idx[num_train:]]
            else:
                picked_input_path = curr_input_path
                picked_target_path = curr_target_path
            
            
            input_path.extend(picked_input_path)
            target_path.extend(picked_target_path)
            
        # for debug mode (args.debug)
        if self.args.debug:
            input_path = input_path[:10]
            target_path = target_path[:10]

        if self.load_mode == 0: # batch data load
            self.input_ = input_path
            self.target_ = target_path
        else: # load_mode == 1
            self.input_ = [np.load(f) for f in input_path]
            self.target_ = [np.load(f) for f in target_path]

    def __len__(self):
        return len(self.target_)

    def __getitem__(self, idx):
        input_img, target_img = self.input_[idx], self.target_[idx]
        if self.load_mode == 0:
            input_img, target_img = np.load(input_img), np.load(target_img)

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
            
        if self.if_FDD():
            # add a transform to stack to 3 channels
            mask = afm()
            input_img_dct = dct2(input_img)
            input_img_lf = idct2(input_img_dct * mask)
            input_img_hf = idct2(input_img_dct * (1-mask))
            input_img = np.stack([input_img, input_img_lf, input_img_hf], axis=0)
            
        if self.mode == 'train' and self.patch_size and (not self.is_validation):
            input_patches, target_patches = get_patch(input_img,
                                                      target_img,
                                                      self.patch_n,
                                                      self.patch_size)
            
            dataitem = to_float32(input_patches, target_patches)
            return dataitem
        else:
            dataitem = to_float32(input_img, target_img)
            return dataitem
        
    def if_FDD(self):
        if isinstance(self.args.model_choice, str):
            return self.args.model_choice in FDD_MODEL_NAMES
        elif isinstance(self.args.model_choice, tuple):
            return self.args.model_choice[0] in FDD_MODEL_NAMES
        else:
            return False

def get_patch(full_input_img, full_target_img, patch_n, patch_size):
    # assert full_input_img.shape == full_target_img.shape
    patch_input_imgs = []
    patch_target_imgs = []
    h, w = full_input_img.shape[-2:]
    new_h, new_w = patch_size, patch_size
    
    tops = np.random.choice(np.arange(h-new_h), patch_n, replace=False)
    lefts= np.random.choice(np.arange(w-new_w), patch_n, replace=False)
    
    for idx in range(patch_n): # default: patch_n = 16 (512 -> 10*64) 10*3*64*64
        patch_input_img = full_input_img[...,tops[idx]:tops[idx]+new_h, lefts[idx]:lefts[idx]+new_w]
        patch_target_img = full_target_img[tops[idx]:tops[idx]+new_h, lefts[idx]:lefts[idx]+new_w]
        patch_input_imgs.append(patch_input_img)
        patch_target_imgs.append(patch_target_img)
    return np.array(patch_input_imgs), np.array(patch_target_imgs)

def to_float32(*arrays):
    return tuple(arr.astype(np.float32) for arr in arrays)
    
def get_fl_loader(args, transform=None, batch_size=None, is_validation=False, client_id=0):
    mode=args.mode if not is_validation else 'validation'
    data_abbr=args.data_abbr     
    yaml_path = args.yaml_path
    batch_size= args.batch_size if mode == 'train' else 1
    num_workers=args.num_workers 
    random_seed=args.random_seed
    in_federation=args.in_federation


    with open(yaml_path, 'r') as f:
        patients_dict = yaml.safe_load(f)
        if data_abbr is not None:
            patients_list = patients_dict[data_abbr] if not is_validation else patients_dict['v'][client_id] # default using chest as validation
        else:
            exp_code=args.exp_code
            test_code=args.test_code
            if mode == 'train' or mode == 'validation':
                patients_list = patients_dict[mode][exp_code]
            if mode == 'test':
                patients_list = patients_dict[mode][exp_code][test_code]
            

    # check a stacked list such as [['a']], iterate all the items
    assert isinstance(patients_list, list), "patients_list should be a stacked list"
    for li in patients_list:
        assert isinstance(li, list), "patients_list should be a stacked list"
    
    fl_datasets = []
    for patients in patients_list:
        fl_datasets.append(LDCTDataset(args, patients, transform, random_seed, in_federation, is_validation))
    
    # if mode == 'test', dataloader should not be shuffled.
    fl_loaders = [DataLoader(dataset=fl_dataset, batch_size=batch_size, shuffle= True if mode =='train' else False, num_workers=num_workers) for fl_dataset in fl_datasets]
    
    return fl_loaders if len(fl_loaders)>1 else fl_loaders[0]
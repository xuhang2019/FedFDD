import os
import datetime
from config.ldct_loader import get_fl_loader
from torch.backends import cudnn
from config.solver import Solver
import argparse
from torch.utils.tensorboard import SummaryWriter

FIG_SUFFIX = datetime.datetime.now().strftime('%m%d_%H%M%S')

def check_path(path:str):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Create path : {}'.format(path))
    else:
        print('Path {} exists'.format(path))

def test(args):
    cudnn.benchmark = True
    args.fig_suffix = FIG_SUFFIX
    if args.exp_code:
        args.save_path = os.path.join(args.save_path, args.exp_code)
    check_path(args.save_path)
    writer = SummaryWriter(log_dir=os.path.join(args.save_path,'summary'))
    if args.result_fig:
        fig_path = os.path.join(args.save_path, f'fig_{args.fig_suffix}')
        check_path(fig_path)
  
    fl_loader = get_fl_loader(args=args, mode=args.mode,
                              exp_code=args.exp_code, 
                              test_code=args.test_code, 
                              load_mode=args.load_mode, 
                              data_path=args.data_path,
                              yaml_path=args.yaml_path,
                              patch_n=None, 
                              patch_size=None, 
                              transform=args.transform, 
                              batch_size=args.batch_size if args.mode=='train' else 1, 
                              num_workers=args.num_workers, 
                              random_seed = args.random_seed, 
                              in_federation=args.in_federation, 
                              data_abbr=args.data_abbr)
    solver = Solver(args, fl_loader)
    print(f"fl_loader info: {fl_loader}")
    
    solver.test()
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--load_mode', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='../data_ct_denoise')
    parser.add_argument('--save_path', type=str, default='./save/') # if set exp_code, this will be changed to save_path/exp_code.
    parser.add_argument('--yaml_path', type=str, default='configs/pt.yaml') 
    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)
    parser.add_argument('--trunc_min', type=float, default=-160.0)
    parser.add_argument('--trunc_max', type=float, default=240.0)
    parser.add_argument('--afm_rl', type=float, default=0.45)
    parser.add_argument('--transform', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--random_seed', type=int, default=626)
    parser.add_argument('--test_ckpt', type=str, default='')
    parser.add_argument('--device', type=str)
    parser.add_argument('--num_workers', type=int, default=7)
    parser.add_argument('--result_fig', action='store_true', help='save result figure')
    parser.add_argument('--fl', action='store_true')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--in_federation', action='store_true', help='control the split of training set')
    parser.add_argument('--half_ft', action='store_true', help='half transfer learning')
    parser.add_argument('--exp_code', type=str, default='061801')
    parser.add_argument('--test_code', type=str, default='1')
    parser.add_argument('--data_abbr', type=str, default='', help='data abbr')
    parser.add_argument('--vdebug', action='store_true', help='validation debug')
    parser.add_argument('--test_model', type=str, default='freq', help='test model, in solver.py')
    args = parser.parse_args()
    
    test(args)


  
  
  

    
    

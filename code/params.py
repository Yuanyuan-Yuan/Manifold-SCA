import os
import json
import torch
import argparse

class Params():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--exp_name', type=str, default='test')
        if os.environ.get('MANIFOLD_SCA') is None:
            default_output = ''
        else:
            default_output = os.path.join(os.environ.get('MANIFOLD_SCA'), 'output') + '/'
        parser.add_argument('--output_root', type=str, default=default_output)
        parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'ChestX-ray', 'SC09', 'Sub-URMP', 'COCO', 'DailyDialog'])
        parser.add_argument('--trace_c', type=int, default=-1)
        parser.add_argument('--trace_w', type=int, default=-1)
        parser.add_argument('--batch_size', type=int, default=50)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--lr', type=float, default=2e-4)
        parser.add_argument('--beta1', type=float, default=0.5)
        parser.add_argument('--nz', type=int, default=512)
        parser.add_argument('--nc', type=int, default=3)
        parser.add_argument('--num_epoch', type=int, default=50)
        parser.add_argument('--test_freq', type=int, default=5)
        parser.add_argument('--side', type=str, default='cacheline', choices=['cacheline', 'cachebank', 'pagetable'])
        parser.add_argument('--cpu', type=str, default='intel', choices=['intel', 'amd'])
        parser.add_argument('--cache', type=str, default='dcache', choices=['dcache', 'icache'])
        parser.add_argument('--image_size', type=int, default=128)
        parser.add_argument('--pad_length', type=int, default=64)
        parser.add_argument('--noise_op', type=str, default='', choices=['', 'shift', 'delete', 'noise', 'zero'])
        parser.add_argument('--noise_k', type=float, default=0)
        parser.add_argument('--noise_pp_op', type=str, default='', choices=['', 'out', 'flip', 'order'])
        parser.add_argument('--noise_pp_k', type=float, default=0)
        parser.add_argument('--n_class', type=int, default=-1)
        parser.add_argument('--lambd', type=float, default=100)
        parser.add_argument('--use_refiner', type=int, default=0, choices=[0, 1])
        parser.add_argument('--embed_dim', type=int, default=300)
        parser.add_argument('--attention_dim', type=int, default=512)
        parser.add_argument('--decoder_dim', type=int, default=512)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--grad_clip', type=float, default=5)
        parser.add_argument('--alpha_c', type=float, default=1)
        parser.add_argument('--lms_w', type=int, default=128)
        parser.add_argument('--lms_h', type=int, default=44)
        parser.add_argument('--max_db', type=float, default=45)
        parser.add_argument('--min_db', type=float, default=-100)
        parser.add_argument('--num_ID', type=int, default=-1)
        parser.add_argument('--num_content', type=int, default=-1)

        self.args = parser.parse_args()
        self.args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.parser = parser
        self.get_data_path('data_path.json')

    def parse(self):
        return self.args

    def get_data_path(self, json_path):
        with open(json_path, 'r') as f:
            data_path = json.load(f)
        root_dir = os.environ.get('MANIFOLD_SCA')
        if root_dir is not None:
            for dataset_name in data_path.keys():
                item = data_path[dataset_name]
                for k in item.keys():
                    if k == 'split':
                        continue
                    item[k] = os.path.join(root_dir, item[k])
        self.args.data_path = data_path

    def save_params(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.args.__dict__, f)

    def print_options(self, params):
        message = ''
        message += '----------------- Params ---------------\n'
        for k, v in sorted(vars(params).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

if __name__ == '__main__':
    p = Params()
    args = p.parse()
    args.batch_size = 1
    p.print_options(args)
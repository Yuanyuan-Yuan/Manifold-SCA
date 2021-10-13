import os
import numpy as np
import argparse
import json
import progressbar

from prep_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--ID', type=int, default=1, help='ID')
args = parser.parse_args()

script = 'mem_access'
pin_out = '%s_%d.out' % (script, int(args.ID))
pin = '../../../pin -t obj-intel64/%s.so -o %s' % (script, pin_out)
print('Pintools: %s' % script)

####################
#       CIFAR      #
####################

pad_length = 256 * 256 * 3
total_num = 1

ROOT = os.environ.get('MANIFOLD_SCA')

libjpeg_path = ROOT + '/target/libjpeg-turbo-build/./tjexample'
npz_dir = ROOT + '/data/CIFAR100/pin/npz/'
raw_dir = ROOT + '/data/CIFAR100/pin/raw/'
input_dir = ROOT +  '/data/CIFAR100/image/'

sub_list = ['train/', 'test/']

make_path(npz_dir)

for sub in sub_list:

    label_dir_list = sorted(os.listdir(input_dir + sub))

    total_img_list = []

    for label_dir in label_dir_list:
        image_names = sorted(os.listdir(input_dir + sub + label_dir))
        total_img_list += [label_dir + '/' + name for name in image_names]
    
    # total_img_list = total_img_list[:500]

    unit_len = int(len(total_img_list) // total_num)

    ID = args.ID - 1
    if ID == total_num - 1:
        img_list = total_img_list[ID*unit_len:]
    else:
        img_list = total_img_list[ID*unit_len:(ID+1)*unit_len]

    make_path(npz_dir + sub)
    
    print('File: ', len(img_list))
    print('Total: ', len(total_img_list))

    progress = progressbar.ProgressBar(widgets=widgets, maxval=len(img_list)).start()
    for i, img in enumerate(img_list):
        progress.update(i + 1)
        # img = 'label/name.jpg'
        img_path = input_dir + sub + img

        prefix = '-'.join(img.split('/')).split('.')[0]
        suffix = '.npz'
        npz_path = npz_dir + sub + prefix + suffix
        raw_path = npz_dir + sub + prefix + '.out'

        os.system('%s -- %s %s %s' % (pin, libjpeg_path, img_path, 'img_output_'+str(args.ID)+'.bmp'))
        # os.system('mv %s %s' % (script + '.out', raw_path))
        raw2npz_full(in_path=pin_out,
                npz_path=npz_path,
                raw_path=raw_path,
                pad_length=pad_length
            )
    progress.finish()
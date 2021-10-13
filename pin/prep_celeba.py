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
#      CelebA      #
####################

pad_length = 256 * 256 * 6
total_num = 1

ROOT = os.environ.get('MANIFOLD_SCA')

libjpeg_path = ROOT + '/target/libjpeg-turbo-build/./tjexample'
npz_dir = ROOT + '/data/CelebA_crop128/pin/npz/'
raw_dir = ROOT + '/data/CelebA_crop128/pin/raw/'
input_dir = ROOT +  '/data/CelebA_crop128/image/'
sub_list = ['train/', 'test/']

make_path(npz_dir)
make_path(raw_dir)

for sub in sub_list:

    total_img_list = sorted(os.listdir(input_dir + sub))
    # total_img_list = total_img_list[:2000]

    unit_len = int(len(total_img_list) // total_num)

    ID = args.ID - 1
    if ID == total_num - 1:
        img_list = total_img_list[ID*unit_len:]
    else:
        img_list = total_img_list[ID*unit_len:(ID+1)*unit_len]

    make_path(npz_dir + sub)
    make_path(raw_dir + sub)
    
    print('File: ', len(img_list))
    print('Total: ', len(total_img_list))

    progress = progressbar.ProgressBar(widgets=widgets, maxval=len(img_list)).start()
    for i, img in enumerate(img_list):
        progress.update(i + 1)
        
        img_path = input_dir + sub + img

        prefix = img.split('.')[0]
        suffix = '.npz'
        npz_path = npz_dir + sub + prefix + suffix
        raw_path = raw_dir + sub + prefix + '.out'
        
        os.system('%s -- %s %s %s' % (pin, libjpeg_path, img_path, 'img_output_'+str(args.ID)+'.bmp'))

        # os.system('cp %s %s' % (script + '.out', raw_path))
        raw2npz_full(in_path=pin_out,
                npz_path=npz_path,
                raw_path=raw_path,
                pad_length=pad_length
            )
    progress.finish()
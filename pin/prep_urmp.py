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

###################
#    Sub-URMP     #
###################

pad_length = 512 * 512 * 8
total_num = 1

ROOT = os.environ.get('MANIFOLD_SCA')

ffmpeg_path = ROOT + '/target/ffmpeg'
npz_dir = ROOT + '/data/SC09/pin/npz/'
raw_dir = ROOT + '/data/SC09/pin/raw/'
input_dir = ROOT +  '/data/SC09/lms/'

sub_list = ['train/', 'validation/']

make_path(npz_dir)
make_path(raw_dir)

for sub in sub_list:
    total_file_list = sorted(os.listdir(input_dir + sub))

    # total_file_list = total_file_list[:2000]

    unit_len = int(len(total_file_list) // total_num)

    ID = args.ID - 1
    if ID == total_num - 1:
        file_list = total_file_list[ID*unit_len:]
    else:
        file_list = total_file_list[ID*unit_len:(ID+1)*unit_len]

    make_path(npz_dir + sub)
    make_path(raw_dir + sub)
    
    print('File: ', len(file_list))
    print('Total: ', len(total_file_list))

    progress = progressbar.ProgressBar(widgets=widgets, maxval=len(file_list)).start()
    for i, file in enumerate(file_list):
        progress.update(i + 1)
        
        input_path = input_dir + sub + file
        
        prefix = file.split('.')[0]
        suffix = '.npz'
        npz_path = npz_dir + sub + prefix + suffix
        raw_path = raw_dir + sub + prefix + '.out'

        os.system('%s -- %s -i %s -ac 1 -ar 4000 -loglevel quiet -y %s' % (pin, ffmpeg_path, input_path, 'audio_output_'+str(args.ID)+'.wav'))
        
        # os.system('cp %s %s' % (script + '.out', raw_path))
        raw2npz_full(in_path=pin_out,
                npz_path=npz_path,
                raw_path=raw_path,
                pad_length=pad_length
            )
    progress.finish()
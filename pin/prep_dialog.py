import os
import numpy as np
import argparse
import json
import progressbar

from prep_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--ID', type=int, default=1, help='ID of multi process')
args = parser.parse_args()

script = 'mem_access'
pin_out = '%s_%d.out' % (script, int(args.ID))
pin = '../../../pin -t obj-intel64/%s.so -o %s' % (script, pin_out)
print('Pintools: %s' % script)

####################
#    DailyDialog   #
####################

pad_length = 128 * 128 * 6
total_num = 1

ROOT = os.environ.get('MANIFOLD_SCA')

shunpell_path = ROOT + '/target/hunspell'
npz_dir = ROOT + '/data/DailyDialog_text/pin/npz/'
raw_dir = ROOT + '/data/DailyDialog_text/pin/raw/'
input_dir = ROOT +  '/data/DailyDialog_text/text/'

# hunspell_dict = '/path/to/hunspell_dict/en_US'
hunspell_dict = '/export/d2/yyuanaq/experiment/text_test/en_US'

sub_list = ['train/', 'test/']

make_path(npz_dir)
make_path(raw_dir)

for sub in sub_list:
    
    input_file_path = input_dir + ('%s.json' % sub[:-1])
    with open(input_file_path, 'r') as f:
        total_sent_list = json.load(f)

    # total_sent_list = total_sent_list[:2000]
    total_name_list = list(range(len(total_sent_list)))

    unit_len = int(len(total_sent_list) // total_num)

    ID = args.ID - 1
    if ID == total_num - 1:
        sent_list = total_sent_list[ID*unit_len:]
        name_list = total_name_list[ID*unit_len:]
    else:
        sent_list = total_sent_list[ID*unit_len:(ID+1)*unit_len]
        name_list = total_name_list[ID*unit_len:(ID+1)*unit_len]

    make_path(npz_dir + sub)
    make_path(raw_dir + sub)
    
    print('File: ', len(sent_list))
    print('Total: ', len(total_sent_list))

    progress = progressbar.ProgressBar(widgets=widgets, maxval=len(sent_list)).start()
    for i, sent in enumerate(sent_list):
        progress.update(i + 1)
        
        prefix = str('%07d' % name_list[i])
        suffix = '.npz'
        npz_path = npz_dir + sub + prefix + suffix
        raw_path = raw_dir + sub + prefix + '.out'

        os.system('%s -- echo "%s" | %s -d %s -l' % (pin, sent, hunspell_path, hunspell_dict))        
        # os.system('cp %s %s' % (script + '.out', raw_path))
        raw2npz_full(in_path=pin_out,
                npz_path=npz_path,
                raw_path=raw_path,
                pad_length=pad_length
            )
    progress.finish()
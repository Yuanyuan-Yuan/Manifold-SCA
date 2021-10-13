import os
import sys
import time
import subprocess
import numpy as np
import json
from tqdm import tqdm

cpu_id = int(sys.argv[1])
seg_id = int(sys.argv[2])
total_num = 4
assert seg_id in list(range(total_num))

ROOT = os.environ.get('MANIFOLD_SCA')

input_dir = ROOT + '/data/DailyDialog_text/text/'

pp_exe = ROOT + '/prime_probe/Mastik/demo/./L1-capture'
side_dir = ROOT + '/data/DailyDialog_text/pp/intel-dcache/'

# pp_exe = ROOT + '/prime_probe/Mastik/demo/./L1i-capture'
# side_dir = ROOT + '/data/DailyDialog_text/pp/intel-icache/'

hunspell_path = ROOT + '/target/./hunspell'

cache_path = ('cache_text_%03d-%03d.txt' % (cpu_id, seg_id))
pp_cmd = ('taskset -c %d %s %s' % (cpu_id, pp_exe, cache_path))

TRY_NUM = 8 # AMD L1
PAD_LEN = 1024 * TRY_NUM # AMD L1

# TRY_NUM = 8 # AMD L1i
# PAD_LEN = 256 * TRY_NUM # AMD L1i

# TRY_NUM = 8 # Intel L1
# PAD_LEN = 512 * TRY_NUM # Intel L1

# TRY_NUM = 8 # Intel L1i
# PAD_LEN = 512 * TRY_NUM # Intel L1i

N_SET = 64

for sub in ['train/', 'test/']:
    
    input_file_path = input_dir + ('%s.json' % sub[:-1])
    with open(input_file_path, 'r') as f:
        total_sent_list = json.load(f)
    total_name_list = list(range(len(total_sent_list)))

    assert len(total_sent_list) == len(total_name_list)

    unit_len = int(len(total_sent_list) // total_num)
    if seg_id == total_num - 1:
        sent_list = total_sent_list[seg_id*unit_len:]
        name_list = total_name_list[seg_id*unit_len:]
    else:
        sent_list = total_sent_list[seg_id*unit_len:(seg_id+1)*unit_len]
        name_list = total_name_list[seg_id*unit_len:(seg_id+1)*unit_len]
    
    for sent_idx in tqdm(range(len(sent_list))):
        sent = sent_list[sent_idx]        
        prefix = str('%07d' % name_list[sent_idx])
        suffix = '.npz'
        side_path = side_dir + sub + prefix + suffix
        victim_cmd = ('taskset -c %d echo "%s" | hunspell -d /export/d2/yyuanaq/experiment/text_test/en_US -l' % (cpu_id, sent))
        
        data_list = []
        for try_idx in range(TRY_NUM):
            pp_proc = subprocess.Popen([pp_cmd], shell=True)
            time.sleep(0.002)

            start_time = time.time()
            os.system(victim_cmd)
            end_time = time.time()

            time.sleep(0.004)
            # os.system("sudo pkill -9 -P " + str(pp_proc.pid))
            os.system("sudo kill -9 " + str(pp_proc.pid))

            # print('delta: ', end_time - start_time)

            with open(cache_path, 'r') as f:
                lines = f.readlines()

            access_list = []
            for l in lines:
                if len(l) > 1:
                    info = l.strip().split(' ')
                    cur = float(info[0])
                    if cur > end_time:
                        break
                    if cur >= start_time:
                        access_list.append(np.array(list(map(int, info[1:]))))

            for i in range(len(access_list) - 1):
                prev = access_list[i]
                nxt = access_list[i + 1]
                miss = (prev == 1) & (nxt == 0)
                if np.sum(miss) > 0:
                    data_list.append(miss.astype(float))
        
        # print(data_list[0].shape)
        # print(np.zeros(N_SET).shape)
        # print(len(data_list))
        if len(data_list) < PAD_LEN:
            for _ in range(PAD_LEN - len(data_list)):
                data_list.append(np.zeros(N_SET))
        else:
            data_list = data_list[:PAD_LEN]
        # print(len(data_list))
        np.savez_compressed(side_path, np.array(data_list))

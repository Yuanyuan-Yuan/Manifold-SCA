import os
import time
import subprocess
import sys
import numpy as np
from tqdm import tqdm

cpu_id = int(sys.argv[1])
seg_id = int(sys.argv[2])
total_num = 8
assert seg_id in list(range(total_num))

ROOT = os.environ.get('MANIFOLD_SCA')

input_dir = ROOT + '/data/CelebA_crop128/image/'

pp_exe = ROOT + '/prime_probe/Mastik/demo/./L1-capture'
side_dir = ROOT + '/data/CelebA_crop128/pp/intel-dcache/'

# pp_exe = ROOT + '/prime_probe/Mastik/demo/./L1i-capture'
# side_dir = ROOT + '/data/CelebA_crop128/pp/intel-icache/'

libjpeg_path = ROOT + '/target/./tjexample'

cache_path = ('cache_image_%03d-%03d.txt' % (cpu_id, seg_id))
pp_cmd = ('taskset -c %d %s %s' % (cpu_id, pp_exe, cache_path))
out_path = ('output_%03d-%03d.bmp' % (cpu_id, seg_id))

# TRY_NUM = 8 # AMD L1
# PAD_LEN = 512 * TRY_NUM # AMD L1

# TRY_NUM = 8 # AMD L1I
# PAD_LEN = 128 * TRY_NUM # AMD L1I

TRY_NUM = 8 # Intel L1
PAD_LEN = 128 * TRY_NUM # Intel L1

# TRY_NUM = 8 # Intel L1i
# PAD_LEN = 128 * TRY_NUM # Intel L1i

N_SET = 64

for sub in ['train/', 'test/']:
    total_image_list = sorted(os.listdir(input_dir + sub))

    unit_len = int(len(total_image_list) // total_num)
    if seg_id == total_num - 1:
        image_list = total_image_list[seg_id*unit_len:]
    else:
        image_list = total_image_list[seg_id*unit_len:(seg_id+1)*unit_len]

    for image_idx in tqdm(range(len(image_list))):
        image_name = image_list[image_idx]
        prefix = image_name.split('.')[0]
        in_path = input_dir + sub + image_name
        side_path = side_dir + sub + prefix + '.npz'
        victim_cmd = ('taskset -c %d %s %s %s' % (cpu_id, libjpeg_path, in_path, out_path))
        data_list = []
        
        miss_cnt = 0

        for try_idx in range(TRY_NUM):
            pp_proc = subprocess.Popen([pp_cmd], shell=True)
            time.sleep(0.002)

            start_time = time.time()
            os.system(victim_cmd)
            # time.sleep(0.02)
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
                try:
                    miss = (prev == 1) & (nxt == 0)
                    if np.sum(miss) > 0:
                        data_list.append(miss.astype(float))
                except:
                    print('Size does not match.')
        
        # print(data_list[0].shape)
        # print(np.zeros(N_SET).shape)
        # print('miss_cnt: ', miss_cnt)
        # print(len(data_list))
        if len(data_list) < PAD_LEN:
            for _ in range(PAD_LEN - len(data_list)):
                data_list.append(np.zeros(N_SET))
        else:
            data_list = data_list[:PAD_LEN]
        # print(len(data_list))
        np.savez_compressed(side_path, np.array(data_list))

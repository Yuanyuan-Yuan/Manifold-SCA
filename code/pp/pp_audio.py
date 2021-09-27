import os
import sys
import time
import subprocess
import numpy as np
from tqdm import tqdm

cpu_id = int(sys.argv[1])
seg_id = int(sys.argv[2])
total_num = 16
assert seg_id in list(range(total_num))

cache_path = ('cache_audio_%03d-%03d.txt' % (cpu_id, seg_id))
fr_cmd = ('taskset -c %d /root/side_channel/Mastik-0.02-AyeAyeCapn/demo/./L1-capture %s' % (cpu_id, cache_path))
out_path = ('output_%03d-%03d.ogg' % (cpu_id, seg_id))

# TRY_NUM = 4 # AMD L1
# PAD_LEN = 8192 * TRY_NUM # AMD L1

# TRY_NUM = 4 # AMD L1i
# PAD_LEN = 2048 * TRY_NUM # AMD L1i

TRY_NUM = 4 # Intel L1
PAD_LEN = 2048 * TRY_NUM # Intel L1

# TRY_NUM = 4 # Intel L1i
# PAD_LEN = 8192 * TRY_NUM # Intel L1i

N_SET = 64

input_dir = '/root/dataset/sc09/'
side_dir = '/root/dataset/sc09_realworld_DCACHE/'

ffmpeg_path = 'ffmpeg'

for sub in ['train/', 'test/']:
    total_audio_list = sorted(os.listdir(input_dir + sub))

    unit_len = int(len(total_audio_list) // total_num)
    if seg_id == total_num - 1:
        audio_list = total_audio_list[seg_id*unit_len:]
    else:
        audio_list = total_audio_list[seg_id*unit_len:(seg_id+1)*unit_len]

    for audio_idx in tqdm(range(len(audio_list))):
        audio_name = audio_list[audio_idx]

        prefix = audio_name.split('.')[0]
        in_path = input_dir + sub + audio_name
        side_path = side_dir + sub + prefix + '.npz'
        victim_cmd = ('taskset -c %d %s -i %s -ac 1 -loglevel quiet -y %s' % (cpu_id, ffmpeg_path, in_path, out_path))
        
        data_list = []
        for try_idx in range(TRY_NUM):
            fr_proc = subprocess.Popen([fr_cmd], shell=True)
            time.sleep(0.002)

            start_time = time.time()
            os.system(victim_cmd)
            end_time = time.time()

            time.sleep(0.004)
            # os.system("sudo pkill -9 -P " + str(fr_proc.pid)) # AMD
            os.system("sudo kill -9 " + str(fr_proc.pid)) # Intel

            with open(cache_path, 'r') as f:
                lines = f.readlines()

            access_list = []
            for l in lines:
                if len(l) > 1:
                    info = l.strip().split(' ')
                    cur = float(info[0])
                    if cur > end_time:
                        # print('Finish.')
                        break
                    if cur >= start_time:
                        access_list.append(np.array(list(map(int, info[1:]))))

            for i in range(len(access_list) - 1):
                prev = access_list[i]
                nxt = access_list[i + 1]
                miss = (prev == 1) & (nxt == 0)
                if np.sum(miss) > 0:
                    data_list.append(miss.astype(float))
        
        if len(data_list) < PAD_LEN:
            for _ in range(PAD_LEN - len(data_list)):
                data_list.append(np.zeros(N_SET))
        else:
            data_list = data_list[:PAD_LEN]
        np.savez_compressed(side_path, np.array(data_list))
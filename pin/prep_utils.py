import os
import numpy as np
import argparse
import json
import progressbar

widgets = ['Progress: ', progressbar.Percentage(), ' ', 
            progressbar.Bar('#'), ' ', 'Count: ', progressbar.Counter(), ' ',
            progressbar.Timer(), ' ', progressbar.ETA()]

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def raw2npz_full(in_path, npz_path, raw_path, pad_length):
    with open(in_path, 'r') as f1:
        lines = f1.readlines()
    mem_arr = []
    with open(raw_path, 'w') as f2:
        for info in lines[:-1]:
            if len(info.split('; ')) != 3:
                print('Error 1.')
                continue
            [lib, rtn, content] = info.split('; ')
            if len(content.split(' ')) != 3:
                print('Error 2.')
                continue
            [ins, op, mem]  = content.split(' ')
            try:
                addr = int(mem, 16)
                if op == 'R':
                    mem_arr.append(addr)
                elif op == 'W':
                    mem_arr.append(-addr)
                else:
                    print('Wrong OP.')
            except:
                print('Wrong Base.')
            f2.write(info)
        print('Length: ', len(mem_arr))
        if len(mem_arr) < pad_length:
            mem_arr += [0] * (pad_length - len(mem_arr))
        else:
            mem_arr = mem_arr[:pad_length]
        # os.system('cp %s %s' % (in_path, raw_path))
        np.savez_compressed(npz_path, np.array(mem_arr))
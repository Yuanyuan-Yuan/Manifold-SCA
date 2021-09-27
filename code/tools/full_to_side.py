import os
import numpy as np
import argparse
import progressbar

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

class AddrToSide(object):

    def __init__(self):
        self.MASK = 0xFFFF_FFFF

    def to_cacheline(self, addr):
        return (addr & self.MASK) >> 6

    def to_pagetable(self, addr):
        return (addr & self.MASK) >> 12 # i.e., addr & (~4095)

    def addr_to_all(self, in_path, cacheline_path, pagetable_path, RW_path):
        file = open(in_path, 'r')
        lines = file.readlines()
        cacheline_arr = []
        pagetable_arr = []
        RW_arr = []
        for content in lines[:-1]: # NOTE: the last line is `#eof`
            [ins, op, mem]  = content.split(' ')
            ##########################################################
            #  [instruction addr, read or write, accessed mem addr]  #
            #  Both `instruction addr` and `accessed mem addr` are   #
            #  in HEX formats, and `read or write` is a single       #
            #  character from {`R`, `W`}.                            #
            ##########################################################
            addr = int(mem, 16)
            cacheline = self.to_cacheline(addr)
            pagetable = self.to_pagetable(addr)
            RW = 1 if op == 'R' else 0
            cacheline_arr.append(cacheline)
            pagetable_arr.append(pagetable)
            RW_arr.append(RW)
        np.savez_compressed(cacheline_path, np.array(cacheline_arr))
        np.savez_compressed(pagetable_path, np.array(pagetable_arr))
        np.savez_compressed(RW_path, np.array(RW_arr))

    def addr_to_cacheline(self, in_path, out_path):
        file = open(in_path, 'r')
        lines = file.readlines()
        cacheline_arr = []
        for content in lines[:-1]:
            [ins, op, mem]  = content.split(' ')
            addr = int(mem, 16)
            cacheline = self.to_cacheline(addr)
            cacheline_arr.append(cacheline)
        np.savez_compressed(out_path, np.array(cacheline_arr))

    def addr_to_pagetable(self, in_path, out_path):
        file = open(in_path, 'r')
        lines = file.readlines()
        pagetable_arr = []
        for content in lines[:-1]:
            [ins, op, mem]  = content.split(' ')
            addr = int(mem, 16)
            pagetable = self.to_pagetable(addr)
            pagetable_arr.append(pagetable)
        np.savez_compressed(out_path, np.array(pagetable_arr))

    def addr_to_RW(self, in_path, out_path):
        file = open(in_path, 'r')
        lines = file.readlines()
        RW_arr = []
        for content in lines[:-1]:
            [ins, op, mem]  = content.split(' ')
            RW = 1 if op == 'R' else 0
            RW_arr.append(RW)
        np.savez_compressed(out_path, np.array(RW_arr))


####################
#      CelebA      #
####################

input_dir = '/root/data/celeba_pin/'

cacheline_dir = '/root/data/celeba_cacheline/'
pagetable_dir = '/root/data/celeba_pagetable/'
RW_dir = '/root/data/celeba_RW/'

sub_list = ['train/', 'val/', 'test/']

make_path(cacheline_dir)
make_path(pagetable_dir)
make_path(RW_dir)

tool = AddrToSide()

for sub in sub_list:

    file_list = sorted(os.listdir(input_dir + sub))

    make_path(cacheline_dir + sub)
    make_path(pagetable_dir + sub)
    make_path(RW_dir + sub)

    progress = progressbar.ProgressBar(maxval=len(file_list)).start()
    for i, file_name in enumerate(file_list):
        progress.update(i + 1)
        ###############################################
        # This operation depends on the FILE_NAME     #
        # set by your pintool. I assume the FILE_NAME #
        # is `XXXX.out`.                              #
        ###############################################
        prefix = file_name.split('.')[0]
        in_path = input_dir + sub + file_name
        cacheline_path = cacheline_dir + sub + prefix + '.npz'
        pagetable_path = pagetable_dir + sub + prefix + '.npz'
        RW_path = RW_dir + sub + prefix + '.npz'

        # tool.addr_to_cacheline(
        #     in_path=npz_path,
        #     out_path=cacheline_path
        #     )
        # tool.addr_to_pagetable(
        #     in_path=npz_path,
        #     out_path=pagetable_path
        #     )
        # tool.addr_to_RW(
        #     in_path=npz_path,
        #     out_path=RW_path
        #     )
        tool.addr_to_all(
            in_path=npz_path,
            cacheline_path=cacheline_path,
            pagetable_path=pagetable_path,
            RW_path=RW_path,
            )

    progress.finish()

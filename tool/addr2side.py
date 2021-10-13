import os
import numpy as np
import argparse
import progressbar

parser = argparse.ArgumentParser()
parser.add_argument('--ID', type=int, default=1, help='ID')
args = parser.parse_args()

widgets = ['Progress: ', progressbar.Percentage(), ' ', 
            progressbar.Bar('#'), ' ', 'Count: ', progressbar.Counter(), ' ',
            progressbar.Timer(), ' ', progressbar.ETA()]

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

class FullToSide(object):

    def __init__(self):
        self.MASK32 = 0xFFFF_FFFF
        self.MASK_ASLR = 0xFFF

    def to_cacheline32(self, addr, ASLR=False):
        if ASLR:
            addr = addr & self.MASK_ASLR
        return (addr & self.MASK32) >> 6

    def to_cachebank32(self, addr, ASLR=False):
        if ASLR:
            addr = addr & self.MASK_ASLR
        return (addr & self.MASK32) >> 2

    def to_pagetable32(self, addr):
        return (addr & self.MASK32) >> 12

    def to_cacheline64(self, addr, ASLR=False):
        if ASLR:
            addr = addr & self.MASK_ASLR
        return addr >> 6

    def to_cachebank64(self, addr, ASLR=False):
        if ASLR:
            addr = addr & self.MASK_ASLR
        return addr >> 2

    def to_pagetable64(self, addr):
        return addr >> 12

    def full_to_all(self, in_path, cacheline_path, pagetable_path, cachebank_path, n_bits=32, ASLR=False):
        full = np.load(in_path)['arr_0']
        cacheline_arr = []
        pagetable_arr = []
        cachebank_arr = []
        for addr in full:
            w = (-1 if addr < 0 else 1)
            addr = abs(addr)
            if n_bits == 32:
                cacheline = self.to_cacheline32(addr, ASLR)
                pagetable = self.to_pagetable32(addr)
                cachebank = self.to_cachebank32(addr, ASLR)
            else:
                cacheline = self.to_cacheline64(addr, ASLR)
                pagetable = self.to_pagetable64(addr)
                cachebank = self.to_cachebank64(addr, ASLR)
            cacheline_arr.append(w * cacheline)
            pagetable_arr.append(w * pagetable)
            cachebank_arr.append(w * cachebank)
        np.savez_compressed(cacheline_path, np.array(cacheline_arr))
        np.savez_compressed(pagetable_path, np.array(pagetable_arr))
        np.savez_compressed(cachebank_path, np.array(cachebank_arr))

    def full_to_cacheline(self, in_path, out_path, n_bits=32, ASLR=False):
        full = np.load(in_path)['arr_0']
        cacheline_arr = []
        for addr in full:
            w = (-1 if addr < 0 else 1)
            addr = abs(addr)
            if n_bits == 32:
                cacheline = self.to_cacheline32(addr, ASLR)
            else:
                cacheline = self.to_cacheline64(addr, ASLR)
            cacheline_arr.append(w * cacheline)
        np.savez_compressed(out_path, np.array(cacheline_arr))

    def full_to_pagetable(self, in_path, out_path, n_bits=32):
        full = np.load(in_path)['arr_0']
        pagetable_arr = []
        for addr in full:
            w = (-1 if addr < 0 else 1)
            addr = abs(addr)
            if n_bits == 32:
                pagetable = self.to_pagetable32(addr)
            else:
                pagetable = self.to_pagetable64(addr)
            pagetable_arr.append(w * pagetable)
        np.savez_compressed(out_path, np.array(pagetable_arr))

    def full_to_cachebank(self, in_path, out_path, n_bits=32, ASLR=False):
        full = np.load(in_path)['arr_0']
        cachebank_arr = []
        for addr in full:
            w = (-1 if addr < 0 else 1)
            addr = abs(addr)
            if n_bits == 32:
                cachebank = self.to_cachebank32(addr, ASLR)
            else:
                cachebank = self.to_cachebank64(addr, ASLR)
            cachebank_arr.append(w * cachebank)
        np.savez_compressed(out_path, np.array(cachebank_arr))

####################
#      CelebA      #
####################

ROOT = os.environ.get('MANIFOLD_SCA')

input_dir = ROOT + '/data/CelebA_crop128/pin/npz/'
total_num = 1

cacheline_dir = ROOT + '/data/SC09/pin/cacheline/'
pagetable_dir = ROOT + '/data/CelebA_crop128/pin/pagetable/'
cachebank_dir = ROOT + '/data/CelebA_crop128/pin/cachebank/'

sub_list = [sub + '/' for sub in sorted(os.listdir(input_dir))]

make_path(cacheline_dir)
make_path(pagetable_dir)
make_path(cachebank_dir)

tool = FullToSide()

for sub in sub_list:

    total_npz_list = sorted(os.listdir(input_dir + sub))

    unit_len = int(len(total_npz_list) // total_num)

    ID = args.ID - 1
    if ID == total_num - 1:
        npz_list = total_npz_list[ID*unit_len:]
    else:
        npz_list = total_npz_list[ID*unit_len:(ID+1)*unit_len]

    make_path(cacheline_dir + sub)
    make_path(pagetable_dir + sub)
    make_path(cachebank_dir + sub)
    
    print('File: ', len(npz_list))
    print('Total: ', len(total_npz_list))

    progress = progressbar.ProgressBar(maxval=len(npz_list), widgets=widgets).start()
    for i, npz_name in enumerate(npz_list):
        progress.update(i + 1)
        
        npz_path = input_dir + sub + npz_name
        cacheline_path = cacheline_dir + sub + npz_name
        pagetable_path = pagetable_dir + sub + npz_name
        cachebank_path = cachebank_dir + sub + npz_name

        # tool.full_to_cacheline(
        #     in_path=npz_path,
        #     out_path=cacheline_path,
        #     n_bits=32,
        #     ASLR=False
        #     )
        # tool.full_to_pagetable(
        #     in_path=npz_path,
        #     out_path=pagetable_path,
        #     n_bits=32
        #     )
        # tool.full_to_cachebank(
        #     in_path=npz_path,
        #     out_path=cachebank_path,
        #     n_bits=32,
        #     ASLR=False
        #     )
        
        tool.full_to_all(
            in_path=npz_path,
            cacheline_path=cacheline_path,
            pagetable_path=pagetable_path,
            cachebank_path=cachebank_path,
            n_bits=32,
            ASLR=False
            )

    progress.finish()

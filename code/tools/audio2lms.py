# audio processing
from scipy import signal # audio processing
from scipy.fftpack import dct
import librosa # library for audio processing
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import librosa.output
import progressbar
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ID', type=int, default=-1, help='ID')
args = parser.parse_args()

def to_lms(audio_path, lms_path):
    audio, sr = librosa.load(audio_path) 
    ps = librosa.feature.melspectrogram(y=audio, sr=sr)
    ps_db = librosa.power_to_db(ps)
    #ps = librosa.feature.mfcc(y=audio, sr=sr)
    #final_pad = np.zeros([128, 44]) # SC09
    final_pad = np.zeros([128, 22]) # Sub-URMP
    final_pad[:ps_db.shape[0],:ps_db.shape[1]] = ps_db
    np.savez_compressed(lms_path, final_pad)

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

widgets = ['Progress: ', progressbar.Percentage(), ' ', 
            progressbar.Bar('#'), ' ', 'Count: ', progressbar.Counter(), ' ',
            progressbar.Timer(), ' ', progressbar.ETA()]

total_num = 5

lms_dir = '/root/dataset/Sub-URMP_raw_lms/'

input_dir = '/root/dataset/Sub-URMP-Audio/'
sub_list = ['train/', 'validation/']

make_path(lms_dir)

for sub in sub_list:
    make_path(lms_dir + sub)

    total_file_list = sorted(os.listdir(input_dir + sub))

    unit_len = int(len(total_file_list) // total_num)

    ID = args.ID - 1
    if ID == total_num - 1:
        file_list = total_file_list[ID*unit_len:]
    else:
        file_list = total_file_list[ID*unit_len:(ID+1)*unit_len]

    print('Total: ', len(total_file_list))
    print('File: ', len(file_list))

    progress = progressbar.ProgressBar(widgets=widgets, maxval=len(file_list)).start()
    for i, file in enumerate(file_list):
        progress.update(i + 1)
        
        prefix = file.split('.')[0]
        suffix = '.npz'

        input_path = input_dir + sub + file
        lms_path = lms_dir + sub + prefix + suffix
        to_lms(audio_path=input_path,
                lms_path=lms_path,
            )
    progress.finish()
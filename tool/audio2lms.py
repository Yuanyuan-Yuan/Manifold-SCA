# audio processing
from scipy import signal # audio processing
from scipy.fftpack import dct
import librosa # library for audio processing
import numpy as np
import librosa.display
import librosa.output
import os
import argparse
import tqdm

def audio_to_lms(dataset, audio_path, lms_path):
    #audio, sr = librosa.load(audio_path, duration=3)
    audio, sr = librosa.load(audio_path, duration=1) 
    ps = librosa.feature.melspectrogram(y=audio, sr=sr)
    ps_db = librosa.power_to_db(ps)
    if dataset == 'SC09':
        final_pad = np.zeros([128, 44]) # SC09
    elif dataset == 'Sub-URMP':
        final_pad = np.zeros([128, 22]) # Sub-URMP
    else:
        raise NotImplementedError
    final_pad[:ps_db.shape[0],:ps_db.shape[1]] = ps_db
    np.savez_compressed(lms_path, final_pad)

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='SC09', choices=['SC09', 'Sub-URMP'])
parser.add_argument('--input_dir', type=str, default='')
parser.add_argument('--output_dir', type=str, default='')
args = parser.parse_args()


make_path(args.output_dir)
name_list = sorted(os.listdir(args.input_dir))
for name in tqdm(name_list):
    prefix = name.split('.')[0]
    input_path = os.path.join(args.input_dir, name)
    output_path = os.path.join(args.output_dir, prefix + '.npz')
    audio_to_lms(
        dataset=args.dataset,
        audio_path=input_path,
        lms_path=output_path
        )
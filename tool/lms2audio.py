# audio processing
from scipy import signal # audio processing
from scipy.fftpack import dct
import librosa # library for audio processing
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import librosa.output
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import os
import tqdm
import argparse

def lms_to_audio(lms_path, audio_path, img_path):
    lms = np.load(lms_path)['arr_0'][0] # [1, 128, 44] SC09
    #print('Shape: ', lms.shape)
    
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    p = librosa.display.specshow(lms, x_axis='s', y_axis='log', cmap=cm.jet, ax=ax)
    fig.savefig(img_path)

    ps_reverse = librosa.db_to_power(lms)
    audio_reverse = librosa.feature.inverse.mel_to_audio(ps_reverse, sr=22050)
    librosa.output.write_wav(audio_path, audio_reverse, sr=22050)

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='')
parser.add_argument('--output_dir', type=str, default='')
args = parser.parse_args()


make_path(args.output_dir)
name_list = sorted(os.listdir(args.input_dir))
for name in tqdm(name_list):
    prefix = name.split('.')[0]
    input_path = os.path.join(args.input_dir, name)
    audio_path = os.path.join(args.output_dir, prefix + '.wav')
    image_path = os.path.join(args.output_dir, prefix + '.png')
    lms_to_audio(
        lms_path=input_path,
        audio_path=audio_path,
        img_path=image_path
        )
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
import progressbar

widgets = ['Progress: ', progressbar.Percentage(), ' ', 
            progressbar.Bar('#'), ' ', 'Count: ', progressbar.Counter(), ' ',
            progressbar.Timer(), ' ', progressbar.ETA()]

def lms_to_audio(lms_path, audio_path, img_path):
    lms = np.load(lms_path)['arr_0'][0] # [1, 128, 44]
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

for dataset in ['Sub-URMP_cacheline']:
    print(dataset)
    lms_dir = ('/root/output/sca_evaluation/%s/recons/' % dataset)
    audio_dir = ('/root/output/sca_evaluation/%s/recons_audio/' % dataset)
    img_dir = ('/root/output/sca_evaluation/%s/recons_img/' % dataset)

    make_path(audio_dir)
    make_path(img_dir)

    lms_list = sorted(os.listdir(lms_dir))

    progress = progressbar.ProgressBar(widgets=widgets, maxval=len(lms_list)).start()
    for i, lms_name in enumerate(lms_list):
        progress.update(i + 1)
        prefix = lms_name.split('.')[0]
        lms_path = lms_dir + lms_name
        audio_path = audio_dir + prefix + '.wav'
        img_path = img_dir + prefix + '.png'
        lms_to_audio(
                lms_path=lms_path,
                audio_path=audio_path,
                img_path=img_path
            )
    progress.finish()


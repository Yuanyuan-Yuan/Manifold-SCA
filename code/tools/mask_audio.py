import os
import librosa # library for audio processing
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import librosa.output

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

dataset = 'Sub-URMP'

op_list = ['noised', 'other', 'dominate1', 'dominate2']

for op in op_list:
    print('op: ', op)
    weight_list = [0.05, 0.1, 0.3]

    in_audio_dir = ('/root/dataset/blinding_append/audio/%s/%s/' % (op, dataset))
    out_audio_dir = ('/root/dataset/blinding_append/%s/%s/' % (op, dataset))

    make_path(out_audio_dir)

    for weight in weight_list:
        print('weight: %f' % weight)
        input_dir = ((in_audio_dir + '%f/') % weight)
        output_dir = ((out_audio_dir + '%f/') % weight)
        make_path(output_dir)
        audio_list = sorted(os.listdir(input_dir))[:1000]
        for audio_name in audio_list:
            input_path = input_dir + audio_name
            output_path = output_dir + audio_name
            os.system('ffmpeg -i %s -ac 1 -ar 2000 -loglevel quiet -y %s' % (input_path, output_path))
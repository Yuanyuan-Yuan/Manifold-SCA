import os
import cv2
import json
import argparse
import numpy as np

import librosa # library for audio processing
import librosa.display
import librosa.output

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def add(media, mask, data, alpha, beta):
    if media == 'image':
        masked = alpha * data + beta * mask
    elif media == 'audio':
        padded = np.zeros(mask.shape)
        padded[:audio.shape[0]] = data
        masked = alpha * data + beta * mask
    elif media == 'text':
        num = 1 / alpha - 1
        word_list = data.split(' ')
        masked_word_list = []
        for word in word_list:
            masked_word_list += ([word] + [mask] * num)
        masked = ' '.join(masked_word_list)
    return masked


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='')
parser.add_argument('--output_dir', type=str, default='')
parser.add_argument('--mask', type=str, default='')
parser.add_argument('--mask_weight', type=float, default=0.9)
parser.add_argument('--media', type=str, default='image', choices=['image', 'audio', 'text'])
args = parser.parse_args()

assert args.mask_weight >= 0 and args.mask_weight <= 1

beta = args.mask_weight
alpha = 1 - beta

if media == 'text':
    with open(args.input_dir, 'r') as f:
        sent_list = json.load(f)

    masked_sent_list = []
    for sent in sent_list:
        masked_sent = add(args.media, args.mask, sent, alpha, beta)
        masked_sent_list.append(masked_sent)

    with open(args.output_dir, 'w') as f:
        json.dump(masked_sent_list, f)

else:
    if args.media == 'image':
        mask_data = cv2.imread(args.mask)
    elif args.media == 'audio':
        mask_data, _ = librosa.load(args.mask)

    make_path(args.output_dir)
    name_list = sorted(os.listdir(args.input_dir))
    for name in name_list:
        input_path = os.path.join(args.input_dir, name)
        output_path = os.path.join(args.output_dir, name)
        
        if args.media == 'image':
            image = cv2.imread(input_path)
            masked_image = add(args.media, mask_data, image, alpha, beta)
            cv2.imwrite(output_path, masked_image)
        
        elif args.media == 'audio':
            audio, sr = librosa.load(input_path)
            masked_audio = add(args.media, mask_data, audio, alpha, beta)
            librosa.output.write_wav(output_path, masked_audio, sr)
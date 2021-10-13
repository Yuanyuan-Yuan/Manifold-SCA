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

def subtract(media, mask, masked, alpha, beta):
    if media in ['image', 'audio']:
        data = (masked - beta * mask) / alpha
    elif media == 'text':
        num = 1 / alpha - 1
        masked_word_list = masked.split(' ')
        word_list = []
        for i, word in enumerate(masked_word_list):
            if i % (num + 1) == 0:
                word_list.append(word)
        data = ' '.join(word_list)
    return data


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
        masked_sent_list = json.load(f)

    sent_list = []
    for masked_sent in masked_sent_list:
        sent = subtract(args.media, args.mask, masked_sent, alpha, beta)
        sent_list.append(sent)

    with open(args.output_dir, 'w') as f:
        json.dump(sent_list, f)

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
            masked_image = cv2.imread(input_path)
            image = subtract(args.media, mask_data, masked_image, alpha, beta)
            cv2.imwrite(output_path, image)
        
        elif args.media == 'audio':
            masked_audio, sr = librosa.load(input_path)
            audio = add(args.media, mask_data, masked_audio, alpha, beta)
            librosa.output.write_wav(output_path, audio, sr)
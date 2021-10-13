import random
import os
import tqdm
import argparse
import numpy as np
import cv2
from skimage.metrics import structural_similarity as SSIM
from skimage.feature import hog
from skimage import exposure
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=100)
parser.add_argument('--K', type=int, default=1)
parser.add_argument('--recons_dir', type=str, default='')
parser.add_argument('--target_dir', type=str, default='')
parser.add_argument('--output_dir', type=str, default='')
args = parser.parse_args()

length = 5000

image_names = sorted(os.listdir(args.recons_dir))[:length]

recons_gray_list = []
target_gray_list = []
for (i, name) in enumerate(tqdm(image_names)):
    path = input_path + name
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        recons_gray_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    path = target_path + name
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        target_gray_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

with open(output_path, 'w') as f:
    T_cnt = 0
    F_cnt = 0
    random.seed(1234)
    for (i, name) in enumerate(tqdm(image_names)):
        N_set = random.sample(range(length), N-1) + [i]
        distance = [SSIM(recons_gray_list[i], target_gray_list[idx]) for idx in N_set]
        idx_distance = np.argsort(-np.array(distance))
        top_set = [N_set[idx_distance[j]] for j in range(K)]
        if i in top_set:
            f.write('%06d\t%d\n' % (i, 1))
            T_cnt += 1
        else:
            f.write('%06d\t%d\n' % (i, 0))
            F_cnt += 1
        if i % 500 == 0:
            print('True: ', T_cnt / (i + 1))
    f.write(str(T_cnt / (i + 1)))
    print(T_cnt / (i + 1))
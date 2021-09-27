import os
import cv2
import numpy as np

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

dataset = 'celeba'
op = 'noised'

weight_list = [0.05, 0.1, 0.3]

mask_path = ('/root/dataset/blinding/mask/%s-%s.bmp' % (dataset, op))

in_image_dir = ('/root/dataset/blinding/processed/%s/%s/' % (op, dataset))

out_image_dir = ('/root/dataset/blinding/recovered/%s/%s/' % (op, dataset))

make_path(out_image_dir)

mask = cv2.imread(mask_path)

for weight in weight_list:
    print('weight: %f' % weight)
    input_dir = ((in_image_dir + '%f/') % weight)
    output_dir = ((out_image_dir + '%f/') % weight)
    make_path(output_dir)
    image_list = sorted(os.listdir(input_dir))
    for image_name in image_list:
        in_path = input_dir + image_name
        in_img = cv2.imread(in_path)
        out_img = (in_img - (1 - weight) * mask) / weight
        out_path = output_dir + image_name
        cv2.imwrite(out_path, out_img)
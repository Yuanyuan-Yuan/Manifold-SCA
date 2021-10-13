import os
import argparse
from tqdm import tqdm
import PIL
from PIL import Image

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def center_crop(img, new_width=128, new_height=128):
    width, height = img.size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    return img.crop((left, top, right, bottom))

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='')
parser.add_argument('--output_dir', type=str, default='')
args = parser.parse_args()


make_path(args.output_dir)
name_list = sorted(os.listdir(args.input_dir))
for name in tqdm(name_list):
    prefix = name.split('.')[0]
    input_path = os.path.join(args.input_dir, name)
    output_path = os.path.join(args.output_dir, (prefix + '.jpg'))
    img = Image.open(input_path)
    out_img = center_crop(img, 128, 128)
    out_img.save(output_path)
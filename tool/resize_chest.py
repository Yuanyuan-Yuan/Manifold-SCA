import os
import argparse
from tqdm import tqdm
import PIL
from PIL import Image

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
    output_path = os.path.join(args.output_dir, (prefix + '.jpg'))
    img = Image.open(input_path)
    out_img = img.resize((32, 32), Image.ANTIALIAS)
    out_img.save(output_path)
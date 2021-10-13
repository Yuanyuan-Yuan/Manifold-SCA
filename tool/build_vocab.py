import os
import json
import argsparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='')
parser.add_argument('--output_path', type=str, default='')
parser.add_argument('--freq', type=int, default=5)
args = parser.parse_args()

with open(args.input_path, 'r') as f:
	sen_list = json.load(f)

word_cnt = {}

for sent in sen_list:
	word_list = sent.strip().split(' ')
	for word in word_list:
		if word in word_cnt.keys():
			word_cnt[word] += 1
		else:
			word_cnt[word] = 1

word_dict = {
	"<PAD>": 0,
	"<UNK>": 1,
	"<START>": 2,
	"<END>": 3
}

for (word, cnt) in word_cnt.items():
	if cnt >= args.freq:
		word_dict[word] = len(word_dict.keys())

with open(output_path, 'w') as f:
	json.dump(word_dict, f)
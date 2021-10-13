import os
import json
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms

import utils


def side_to_bound(side):
    if side == 'cacheline':
        v_max = 0xFFFF_FFFF >> 6
        v_min = -(0xFFFF_FFFF >> 6)
    elif side == 'pagetable':
        v_max = 0xFFFF_FFFF >> 12
        v_min = -(0xFFFF_FFFF >> 12)
    elif side == 'cachebank':
        v_max = 0xFFFF_FFFF >> 2
        v_min = -(0xFFFF_FFFF >> 2)
    else:
        raise NotImplementedError
    return v_max, v_min

class CelebaDataset(Dataset):
    def __init__(self, npz_dir, img_dir, ID_path, split,
                image_size, side, trace_c, trace_w,
                op=None, k=None):
        super(CelebaDataset).__init__()
        self.npz_dir = ('%s%s/' % (npz_dir, split))
        self.img_dir = ('%s%s/' % (img_dir, split))
        self.trace_c = trace_c
        self.trace_w = trace_w
        self.op = op
        self.k = k

        self.npz_list = sorted(os.listdir(self.npz_dir))
        self.img_list = sorted(os.listdir(self.img_dir))

        self.transform = transforms.Compose([
                       transforms.Resize(image_size),
                       transforms.CenterCrop(image_size),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
               ])

        self.v_max, self.v_min = side_to_bound(side)
        
        print('Total %d Data Points.' % len(self.npz_list))

        with open(ID_path, 'r') as f:
            self.ID_dict = json.load(f)

        self.ID_cnt = len(set(self.ID_dict.values()))
        print('Total %d ID.' % self.ID_cnt)

    def __len__(self):
        return len(self.npz_list)

    def __getitem__(self, index):
        npz_name = self.npz_list[index]
        prefix = npz_name.split('.')[0]
        # prefix = int('-'.join(npz_name.split('-')[1:]).split('.')[0])    
        suffix = '.jpg'
        img_name = prefix + suffix
        ID = int(self.ID_dict[img_name]) - 1

        npz = np.load(self.npz_dir + npz_name)
        trace = npz['arr_0']
        trace = trace.astype(np.float32)

        if self.op == 'shift':
            trace = np.concatenate([trace[self.k:], trace[:self.k]])

        if self.op == 'delete':
            assert self.k < 1
            length = len(trace)
            del_num = int(length * self.k)
            del_index = np.random.choice(np.arange(length), del_num, replace=False)
            del_trace = np.delete(trace, del_index)
            trace = np.concatenate([del_trace, np.array([0] * del_num)])
            trace = trace.astype(np.float32)

        trace = torch.from_numpy(trace).view([self.trace_c, self.trace_w, self.trace_w])
        trace = utils.my_scale(v=trace,
                               v_max=self.v_max,
                               v_min=self.v_min
                            )
        if self.op == 'noise':
            trace = (1 - self.k) * trace + self.k * torch.randn(trace.size())

        if self.op == 'zero':
            trace = F.dropout(trace, p=self.k)

        image = Image.open(self.img_dir + img_name)
        image = self.transform(image)

        ID = torch.LongTensor([ID]).squeeze()
        return trace, image, prefix, ID

class ChestDataset(Dataset):
    def __init__(self, npz_dir, img_dir, split, 
                image_size, side, trace_c, trace_w,
                op=None, k=None):
        super(ChestDataset).__init__()
        self.npz_dir = ('%s%s/' % (npz_dir, split))
        self.img_dir = ('%s%s/' % (img_dir, split))
        self.trace_c = trace_c
        self.trace_w = trace_w
        self.op = op
        self.k = k

        self.npz_list = sorted(os.listdir(self.npz_dir))
        self.img_list = sorted(os.listdir(self.img_dir))

        self.transform = transforms.Compose([
                        transforms.Resize(image_size),
                        transforms.CenterCrop(image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

        self.v_max, self.v_min = side_to_bound(side)
        
        print('Total %d Data Points.' % len(self.npz_list))

    def __len__(self):
        return len(self.npz_list)

    def __getitem__(self, index):
        npz_name = self.npz_list[index]
        prefix = npz_name.split('.')[0]
        # prefix = ('-'.join(npz_name.split('-')[1:]).split('.')[0])    
        suffix = '.jpg'
        img_name = prefix + suffix

        npz = np.load(self.npz_dir + npz_name)
        trace = npz['arr_0']
        trace = trace.astype(np.float32)

        if self.op == 'shift':
            trace = np.concatenate([trace[self.k:], trace[:self.k]])

        if self.op == 'delete':
            assert self.k < 1
            length = len(trace)
            del_num = int(length * self.k)
            del_index = np.random.choice(np.arange(length), del_num, replace=False)
            del_trace = np.delete(trace, del_index)
            trace = np.concatenate([del_trace, np.array([0] * del_num)])
            trace = trace.astype(np.float32)

        trace = torch.from_numpy(trace).view([self.trace_c, self.trace_w, self.trace_w])
        trace = utils.my_scale(v=trace,
                               v_max=self.v_max,
                               v_min=self.v_min
                            )

        if self.op == 'noise':
            trace = (1 - self.k) * trace + self.k * torch.randn(trace.size())

        if self.op == 'zero':
            trace = F.dropout(trace, p=self.k)

        image = Image.open(self.img_dir + img_name)
        image = self.transform(image)
        return trace, image, prefix

class DailyDialogDataset(Dataset):
    def __init__(self, text_dir, npz_dir, split, dict_path,
                side, pad_length, trace_c, trace_w, 
                op=None, k=None):
        super(DailyDialogDataset).__init__()
        self.npz_dir = ('%s%s/' % (npz_dir, split))
        data_json_path = ('%s%s.json' % (text_dir, split))
        with open(data_json_path, 'r') as f:
            self.sent_list = json.load(f)

        self.trace_c = trace_c
        self.trace_w = trace_w
        self.op = op
        self.k = k

        def sort_key(npz_name):
            # prefix = int('-'.join(npz_name.split('-')[1:]).split('.')[0])
            prefix = npz_name.split('.')[0]
            return int(prefix)
        self.npz_list = sorted(os.listdir(self.npz_dir), key=sort_key)

        self.v_max, self.v_min = side_to_bound(side)

        # self.npz_list = self.npz_list[:2000]

        with open(dict_path, 'r') as f:
            self.word_dict = json.load(f)

        self.index_dict = {}
        for word in self.word_dict.keys():
            index = self.word_dict[word]
            self.index_dict[index] = word

        self.indexes_list = []
        self.sent_length_list = []
        for npz_name in self.npz_list:
            prefix = int(npz_name.split('.')[0])
            # prefix = int('-'.join(npz_name.split('-')[1:]).split('.')[0])
            sent = self.sent_list[prefix]
            self.sent_length_list.append(min(len(sent.strip().split(' '))+2, pad_length))
        #####################################
        #  sent_len = len(sent.split(' '))  #
        #####################################

        for npz_name in self.npz_list:
            prefix = int(npz_name.split('.')[0])
            # prefix = int('-'.join(npz_name.split('-')[1:]).split('.')[0])
            sent = self.sent_list[prefix]
            self.indexes_list.append(self.sent_to_indexes(
                                        sent=sent,
                                        pad_length=pad_length))

        print('Vocab Size: %d' % len(self.word_dict.keys()))
        print('Max Sentence Length: %d' % max(self.sent_length_list))
        print('Number of Data: %d' % len(self.npz_list))
        print('Number of Sentences: %d' % len(self.sent_list))

    def word_to_index(self, word):
        if word not in self.word_dict.keys():
            return self.word_dict['<UNK>']
        return self.word_dict[word]

    def index_to_word(self, index):
        return self.index_dict[index]

    def sent_to_indexes(self, sent, pad_length=20):
        word_list = sent.strip().split(' ')
        indexes = []
        for word in word_list:
            indexes.append(self.word_to_index(word))
        indexes = [self.word_to_index('<START>')] + \
                  indexes + \
                  [self.word_to_index('<END>')]
        if len(indexes) < pad_length:
            indexes += [self.word_to_index('<PAD>')] * \
                       (pad_length - len(indexes))
        else:
            indexes = indexes[:pad_length-1] + \
                      [self.word_to_index('<END>')]
        return indexes

    def __len__(self):
        return len(self.npz_list)

    def __getitem__(self, index):
        npz_name = self.npz_list[index]
        npz = np.load(self.npz_dir + npz_name)
        trace = npz['arr_0']
        trace = trace.astype(np.float32)

        if self.op == 'shift':
            trace = np.concatenate([trace[self.k:], trace[:self.k]])

        if self.op == 'delete':
            assert self.k < 1
            length = len(trace)
            del_num = int(length * self.k)
            del_index = np.random.choice(np.arange(length), del_num, replace=False)
            del_trace = np.delete(trace, del_index)
            trace = np.concatenate([del_trace, np.array([0] * del_num)])
            trace = trace.astype(np.float32)
            
        trace = torch.from_numpy(trace).view([self.trace_c, self.trace_w, self.trace_w])
        trace = utils.my_scale(v=trace,
                               v_max=self.v_max,
                               v_min=self.v_min
                            )

        if self.op == 'noise':
            trace = (1 - self.k) * trace + self.k * torch.randn(trace.size())

        if self.op == 'zero':
            trace = F.dropout(trace, p=self.k)

        indexes = self.indexes_list[index]
        indexes = np.array(indexes)
        indexes = torch.from_numpy(indexes).type(torch.LongTensor)

        sent_length = self.sent_length_list[index]
        sent_length = torch.LongTensor([sent_length])#.squeeze()

        # prefix = int('-'.join(npz_name.split('-')[1:]).split('.')[0])
        prefix = int(npz_name.split('.')[0])

        return trace, indexes, sent_length, prefix

class CaptionDataset(Dataset):
    def __init__(self, text_dir, npz_dir, split, dict_path,
                side, pad_length, trace_c, trace_w,
                op=None, k=None):
        super(CaptionDataset).__init__()
        self.npz_dir = ('%s%s/' % (npz_dir, split))
        data_json_path = ('%s%s.json' % (text_dir, split))
        with open(data_json_path, 'r') as f:
            self.sent_list = json.load(f)

        self.trace_c = trace_c
        self.trace_w = trace_w
        self.op = op
        self.k = k

        def sort_key(npz_name):
            prefix = int(npz_name.split('.')[0])
            # prefix = int('-'.join(npz_name.split('-')[1:]).split('.')[0])
            return int(prefix)
        self.npz_list = sorted(os.listdir(self.npz_dir), key=sort_key)

        self.v_max, self.v_min = side_to_bound(side)

        # self.npz_list = self.npz_list[:2000]

        with open(dict_path, 'r') as f:
            self.word_dict = json.load(f)

        self.index_dict = {}
        for word in self.word_dict.keys():
            index = self.word_dict[word]
            self.index_dict[index] = word

        self.indexes_list = []
        self.sent_length_list = []
        for npz_name in self.npz_list:
            prefix = int(npz_name.split('.')[0])
            # prefix = int('-'.join(npz_name.split('-')[1:]).split('.')[0])
            sent = self.sent_list[prefix]
            self.sent_length_list.append(min(len(sent.strip().split(' '))+2, pad_length))
        #####################################
        #  sent_len = len(sent.split(' '))  #
        #####################################

        for npz_name in self.npz_list:
            prefix = int(npz_name.split('.')[0])
            # prefix = int('-'.join(npz_name.split('-')[1:]).split('.')[0])
            sent = self.sent_list[prefix]
            self.indexes_list.append(self.sent_to_indexes(
                                        sent=sent,
                                        pad_length=pad_length))

        print('Vocab Size: %d' % len(self.word_dict.keys()))
        print('Max Sentence Length: %d' % max(self.sent_length_list))
        print('Number of Data: %d' % len(self.npz_list))
        print('Number of Sentences: %d' % len(self.sent_list))

    def word_to_index(self, word):
        if word not in self.word_dict.keys():
            return self.word_dict['<UNK>']
        return self.word_dict[word]

    def index_to_word(self, index):
        return self.index_dict[index]

    def sent_to_indexes(self, sent, pad_length=20):
        word_list = sent.strip().split(' ')
        indexes = []
        for word in word_list:
            indexes.append(self.word_to_index(word))
        indexes = [self.word_to_index('<START>')] + \
                  indexes + \
                  [self.word_to_index('<END>')]
        if len(indexes) < pad_length:
            indexes += [self.word_to_index('<PAD>')] * \
                       (pad_length - len(indexes))
        else:
            indexes = indexes[:pad_length-1] + \
                      [self.word_to_index('<END>')]
        return indexes

    def __len__(self):
        return len(self.npz_list)

    def __getitem__(self, index):
        npz_name = self.npz_list[index]
        npz = np.load(self.npz_dir + npz_name)
        trace = npz['arr_0']
        trace = trace.astype(np.float32)

        if self.op == 'shift':
            trace = np.concatenate([trace[self.k:], trace[:self.k]])

        if self.op == 'delete':
            assert self.k < 1
            length = len(trace)
            del_num = int(length * self.k)
            del_index = np.random.choice(np.arange(length), del_num, replace=False)
            del_trace = np.delete(trace, del_index)
            trace = np.concatenate([del_trace, np.array([0] * del_num)])
            trace = trace.astype(np.float32)

        trace = torch.from_numpy(trace).view([self.trace_c, self.trace_w, self.trace_w])
        trace = utils.my_scale(v=trace,
                               v_max=self.v_max,
                               v_min=self.v_min
                            )

        if self.op == 'noise':
            trace = (1 - self.k) * trace + self.k * torch.randn(trace.size())

        if self.op == 'zero':
            trace = F.dropout(trace, p=self.k)

        indexes = self.indexes_list[index]
        indexes = np.array(indexes)
        indexes = torch.from_numpy(indexes).type(torch.LongTensor)

        sent_length = self.sent_length_list[index]
        sent_length = torch.LongTensor([sent_length])#.squeeze()

        # prefix = int('-'.join(npz_name.split('-')[1:]).split('.')[0])
        prefix = int(npz_name.split('.')[0])

        return trace, indexes, sent_length, prefix

class SC09Dataset(Dataset):
    def __init__(self, lms_dir, npz_dir, split,
                side, trace_c, trace_w,
                op=None, k=None,
                max_db=50, min_db=-90):
        super(SC09Dataset).__init__()
        self.lms_dir = ('%s%s/' % (lms_dir, split))
        self.npz_dir = ('%s%s/' % (npz_dir, split))
        self.trace_c = trace_c
        self.trace_w = trace_w
        self.op = op
        self.k = k

        self.lms_list = sorted(os.listdir(self.lms_dir))
        self.npz_list = sorted(os.listdir(self.npz_dir))

        self.v_max, self.v_min = side_to_bound(side)

        self.content_label = {}
        self.ID_label = {}

        self.content_cnt = 0
        self.ID_cnt = 0

        for file_name in self.lms_list:
            content, ID = file_name.split('_')[:2]
            if content not in self.content_label.keys():
                self.content_label[content] = self.content_cnt
                self.content_cnt += 1
            if ID not in self.ID_label.keys():
                self.ID_label[ID] = self.ID_cnt
                self.ID_cnt += 1

        assert self.content_cnt == len(self.content_label.keys())
        assert self.ID_cnt == len(self.ID_label.keys())

        # self.max_db = -10000
        # self.min_db = 10000
        # for lms_name in self.lms_list:
        #     lms = np.load(self.lms_dir + lms_name)
        #     audio = lms['arr_0']
        #     self.max_db = max(self.max_db, np.max(audio))
        #     self.min_db = min(self.min_db, np.min(audio))
        # print('Max db: %f' % self.max_db)
        # print('Min db: %f' % self.min_db)

        self.max_db = max_db
        self.min_db = min_db
        self.norm = transforms.Normalize((0.5,), (0.5,))

        if split == 'train':
            print('Number of Persons (%s): %d' % (split, len(self.ID_label.keys())))
        print('Number of Data Points (%s): %d' % (split, len(self.npz_list)))

    def __len__(self):
        return len(self.npz_list)

    def __getitem__(self, index):
        npz_name = self.npz_list[index]
        npz = np.load(self.npz_dir + npz_name)
        trace = npz['arr_0']
        trace = trace.astype(np.float32)

        if self.op == 'shift':
            trace = np.concatenate([trace[self.k:], trace[:self.k]])

        if self.op == 'delete':
            assert self.k < 1
            length = len(trace)
            del_num = int(length * self.k)
            del_index = np.random.choice(np.arange(length), del_num, replace=False)
            del_trace = np.delete(trace, del_index)
            trace = np.concatenate([del_trace, np.array([0] * del_num)])
            trace = trace.astype(np.float32)

        trace = torch.from_numpy(trace).view([self.trace_c, self.trace_w, self.trace_w]) # upsample
        trace = utils.my_scale(v=trace,
                               v_max=self.v_max,
                               v_min=self.v_min
                            )

        if self.op == 'noise':
            trace = (1 - self.k) * trace + self.k * torch.randn(trace.size())

        if self.op == 'zero':
            trace = F.dropout(trace, p=self.k)

        lms_name = self.lms_list[index]
        # assert '-'.join(npz_name.split('-')[1:]).split('.')[0] == lms_name.split('.')[0]
        assert npz_name.split('.')[0] == lms_name.split('.')[0]
        lms = np.load(self.lms_dir + lms_name)
        audio = lms['arr_0']
        audio = audio.astype(np.float32)
        audio = torch.from_numpy(audio).view([1, 128, 44]) # SC09
        audio = utils.my_scale(v=audio,
                               v_max=self.max_db,
                               v_min=self.min_db
                            )
        audio = self.norm(audio)

        prefix = npz_name.split('.')[0]
        # prefix = '-'.join(npz_name.split('-')[1:])
        content_name, ID_name = prefix.split('_')[:2]
        content = int(self.content_label[content_name])
        ID = int(self.ID_label[ID_name])
        content = torch.LongTensor([content])
        ID = torch.LongTensor([ID])

        return trace, audio, lms_name.split('.')[0], content.squeeze(), ID.squeeze()

class URMPDataset(Dataset):
    def __init__(self, lms_dir, npz_dir, split,
                side, trace_c, trace_w,
                op=None, k=None,
                max_db=50, min_db=-90):
        super(URMPDataset).__init__()
        self.lms_dir = ('%s%s/' % (lms_dir, split))
        self.npz_dir = ('%s%s/' % (npz_dir, split))
        self.trace_c = trace_c
        self.trace_w = trace_w
        self.op = op
        self.k = k

        self.lms_list = sorted(os.listdir(self.lms_dir))
        self.npz_list = sorted(os.listdir(self.npz_dir))

        # self.lms_list = self.lms_list[:1000]
        # self.npz_list = self.npz_list[:1000]

        self.v_max, self.v_min = side_to_bound(side)

        self.content_label = {}
        self.ID_label = {}

        self.content_cnt = 0
        self.ID_cnt = 0

        for file_name in self.lms_list:
            ID = file_name.split('_')[0]
            content = ID[:-2]
            if content not in self.content_label.keys():
                self.content_label[content] = self.content_cnt
                self.content_cnt += 1
            if ID not in self.ID_label.keys():
                self.ID_label[ID] = self.ID_cnt
                self.ID_cnt += 1

        assert self.content_cnt == len(self.content_label.keys())
        assert self.ID_cnt == len(self.ID_label.keys())

        # self.max_db = -10000
        # self.min_db = 10000
        # for lms_name in self.lms_list:
        #     lms = np.load(self.lms_dir + lms_name)
        #     audio = lms['arr_0']
        #     self.max_db = max(self.max_db, np.max(audio))
        #     self.min_db = min(self.min_db, np.min(audio))
        # print('Max db: %f' % self.max_db)
        # print('Min db: %f' % self.min_db)
        
        self.max_db = max_db
        self.min_db = min_db
        
        self.norm = transforms.Normalize((0.5,), (0.5,))

        if split == 'train':
            print('Number of Persons (%s): %d' % (split, len(self.ID_label.keys())))
        print('Number of Data Points (%s): %d' % (split, len(self.npz_list)))

    def __len__(self):
        return len(self.npz_list)

    def __getitem__(self, index):
        npz_name = self.npz_list[index]
        npz = np.load(self.npz_dir + npz_name)
        trace = npz['arr_0']
        trace = trace.astype(np.float32)
        trace = torch.from_numpy(trace).view([self.trace_c, self.trace_w, self.trace_w]) # upsample
        trace = utils.my_scale(v=trace,
                               v_max=self.v_max,
                               v_min=self.v_min
                            )

        lms_name = self.lms_list[index]
        # print('npz_name: ', npz_name)
        # print('lms_name: ', lms_name)
        # assert '-'.join(npz_name.split('-')[1:]).split('.')[0] == lms_name.split('.')[0]
        assert npz_name.split('.')[0] == lms_name.split('.')[0]
        lms = np.load(self.lms_dir + lms_name)
        audio = lms['arr_0']
        audio = audio.astype(np.float32)

        if self.op == 'shift':
            trace = np.concatenate([trace[self.k:], trace[:self.k]])

        if self.op == 'delete':
            assert self.k < 1
            length = len(trace)
            del_num = int(length * self.k)
            del_index = np.random.choice(np.arange(length), del_num, replace=False)
            del_trace = np.delete(trace, del_index)
            trace = np.concatenate([del_trace, np.array([0] * del_num)])
            trace = trace.astype(np.float32)

        audio = torch.from_numpy(audio).view([1, 128, 22]) # Sub-URMP
        audio = utils.my_scale(v=audio,
                               v_max=self.max_db,
                               v_min=self.min_db
                            )

        if self.op == 'noise':
            trace = (1 - self.k) * trace + self.k * torch.randn(trace.size())

        if self.op == 'zero':
            trace = F.dropout(trace, p=self.k)

        audio = self.norm(audio)
        
        ID_name = lms_name.split('_')[0]
        content_name = ID_name[:-2]
        content = int(self.content_label[content_name])
        ID = int(self.ID_label[ID_name])
        content = torch.LongTensor([content])
        ID = torch.LongTensor([ID])

        return trace, audio, lms_name.split('.')[0], content.squeeze(), ID.squeeze()

class CifarDataset(Dataset):
    def __init__(self, image_dir, npz_dir, split,
        image_size, side, trace_c, trace_w):
        super(CifarDataset).__init__()
        self.lms_dir = ('%s%s/' % (lms_dir, split))
        self.npz_dir = ('%s%s/' % (npz_dir, split))
        self.trace_c = trace_c
        self.trace_w = trace_w
        self.npz_list = sorted(os.listdir(self.npz_dir))
        self.transform = transforms.Compose([
                        transforms.Resize(image_size),
                        transforms.CenterCrop(image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
        self.v_max, self.v_min = side_to_bound(side)

    def __len__(self):
        return len(self.npz_list)

    def __getitem__(self, index):
        npz_name = self.npz_list[index]
        prefix = '.'.join(npz_name.split('.')[:-1])
        [label, image_name] = prefix.split('-')
        suffix = '.jpg'

        npz = np.load(self.npz_dir + npz_name)
        trace = npz['arr_0']
        trace = trace.astype(np.float32)
        trace = torch.from_numpy(trace).view([self.trace_c, self.trace_w, self.trace_w])
        trace = utils.my_scale(v=trace,
                               v_max=self.v_max,
                               v_min=self.v_min
                            )
        
        image = Image.open(self.image_dir + label + '/' + image_name + suffix)
        image = self.transform(image)
        return trace, image, prefix


class DataLoader(object):
    def __init__(self, args):
        self.args = args
        self.init_param()
        #self.init_dataset()

    def init_param(self):
        self.gpus = torch.cuda.device_count()
        self.gpus = max(1, self.gpus)
        # self.transform = transforms.Compose([
        #                         transforms.Resize(self.args.image_size),
        #                         transforms.ToTensor(),
        #                         transforms.Normalize((0.5,), (0.5,)),
        #                    ])

    def get_loader(self, dataset, shuffle=True):
        data_loader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=self.args.batch_size * self.gpus,
                            num_workers=int(self.args.num_workers),
                            shuffle=shuffle
                        )
        return data_loader

if __name__ == '__main__':
    pass
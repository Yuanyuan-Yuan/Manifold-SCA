import os
import time
import numpy as np
from PIL import Image
import progressbar
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence

import utils
import models

class RealSideDataset(Dataset):
    def __init__(self, args, split):
        super(RealSideDataset).__init__()
        self.trace_c = args.trace_c
        self.trace_w = args.trace_w
        self.args = args
        self.npz_dir = ('%s%s/' % (args.data_path[args.dataset]['pp-%s-%s' % (args.cpu, args.cache)], split))
        pad_length = args.pad_length
        dict_path = args.data_path[args.dataset]['vocab_path']
        
        text_dir = args.data_path[args.dataset]['media']
        data_json_path = ('%s%s.json' % (text_dir, split))
        with open(data_json_path, 'r') as f:
            self.sent_list = json.load(f)

        def sort_key(npz_name):
            prefix = int(npz_name.split('.')[0])
            return int(prefix)
        self.npz_list = sorted(os.listdir(self.npz_dir), key=sort_key)

        with open(dict_path, 'r') as f:
            self.word_dict = json.load(f)

        self.index_dict = {}
        for word in self.word_dict.keys():
            index = self.word_dict[word]
            self.index_dict[index] = word

        self.indexes_list = []
        self.sent_length_list = []
        for npz_name in self.npz_list:
            #prefix = int(npz_name.split('.')[0])
            prefix = int(npz_name.split('.')[0])
            sent = self.sent_list[prefix]
            self.sent_length_list.append(min(len(sent.strip().split(' '))+2, pad_length))
        #####################################
        #  sent_len = len(sent.split(' '))  #
        #####################################

        for npz_name in self.npz_list:
            #prefix = int(npz_name.split('.')[0])
            prefix = int(npz_name.split('.')[0])
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
        trace = torch.from_numpy(trace).view([self.trace_c, self.trace_w, self.trace_w])

        indexes = self.indexes_list[index]
        indexes = np.array(indexes)
        indexes = torch.from_numpy(indexes).type(torch.LongTensor)

        sent_length = self.sent_length_list[index]
        sent_length = torch.LongTensor([sent_length])#.squeeze()

        prefix = int(npz_name.split('.')[0])

        return trace, indexes, sent_length, prefix

class NoisyRealSideDataset(Dataset):
    def __init__(self, args, split):
        super(NoisyRealSideDataset).__init__()
        self.op = args.noise_pp_op
        self.k = args.noise_pp_k
        self.trace_c = args.trace_c
        self.trace_w = args.trace_w
        self.args = args
        self.npz_dir = args.npz_dir + ('train/' if split == 'train' else 'test/')
        pad_length = args.pad_length
        dict_path = args.vocab_path
        data_json_path = args.data_json_dir + ('dialogues_train.json' if split == 'train' else 'dialogues_test.json')
        with open(data_json_path, 'r') as f:
            self.sent_list = json.load(f)

        def sort_key(npz_name):
            prefix = int(npz_name.split('.')[0])
            return int(prefix)
        self.npz_list = sorted(os.listdir(self.npz_dir), key=sort_key)

        with open(dict_path, 'r') as f:
            self.word_dict = json.load(f)

        self.index_dict = {}
        for word in self.word_dict.keys():
            index = self.word_dict[word]
            self.index_dict[index] = word

        self.indexes_list = []
        self.sent_length_list = []
        for npz_name in self.npz_list:
            #prefix = int(npz_name.split('.')[0])
            prefix = int(npz_name.split('.')[0])
            sent = self.sent_list[prefix]
            self.sent_length_list.append(min(len(sent.strip().split(' '))+2, pad_length))
        #####################################
        #  sent_len = len(sent.split(' '))  #
        #####################################

        for npz_name in self.npz_list:
            #prefix = int(npz_name.split('.')[0])
            prefix = int(npz_name.split('.')[0])
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

        if self.op == 'order':
            assert self.k > 1
            length = len(trace)
            exchange_index = np.random.choice(np.arange(length), self.k, replace=False)
            for ex_i in range(len(exchange_index)-1):
                ex_j = ex_i + 1
                trace[[ex_i, ex_j]] = trace[[ex_j, ex_i]]
        if self.op == 'out':
            assert self.k < 1
            (length, n_set) = trace.shape
            del_num = int(length * self.k)
            del_index = np.random.choice(np.arange(length), del_num, replace=False)
            del_trace = np.delete(trace, del_index, axis=0)
            trace = np.concatenate([del_trace, np.zeros((del_num, n_set))])
            trace = trace.astype(np.float32)

        trace = torch.from_numpy(trace).view([self.trace_c, self.trace_w, self.trace_w])

        if self.op == 'flip':
            assert self.k < 1
            fliped = 1 - trace
            flip_mask = torch.ones(trace.size())
            flip_mask = F.dropout(flip_mask, p=(1-self.k))
            keep_mask = 1 - flip_mask
            trace = flip_mask * fliped + keep_mask * trace

        indexes = self.indexes_list[index]
        indexes = np.array(indexes)
        indexes = torch.from_numpy(indexes).type(torch.LongTensor)

        sent_length = self.sent_length_list[index]
        sent_length = torch.LongTensor([sent_length])#.squeeze()

        prefix = int(npz_name.split('.')[0])

        return trace, indexes, sent_length, prefix


class TextEngine(object):
    def __init__(self, args, idx2word, word2idx):
        self.args = args
        self.epoch = 0
        self.mse = nn.MSELoss().to(self.args.device)
        self.l1 = nn.L1Loss().to(self.args.device)
        self.bce = nn.BCELoss().to(self.args.device)
        self.ce = nn.CrossEntropyLoss().to(self.args.device)
        self.last_output = None
        self.idx2word = idx2word # function
        self.word2idx = word2idx
        self.init_model_optimizer()

    def init_model_optimizer(self):
        print('Initializing Model and Optimizer...')
        self.enc = models.__dict__['trace_encoder_square_%s' % self.args.trace_w](dim=self.args.nz, nc=args.trace_c)
        self.enc = self.enc.to(self.args.device)

        self.dec = models.__dict__['DecoderWithAttention'](
                        attention_dim=self.args.attention_dim, 
                        embed_dim=self.args.embed_dim, 
                        decoder_dim=self.args.decoder_dim,
                        vocab_size=self.args.vocab_size,
                        encoder_dim=self.args.nz,
                        dropout=self.args.dropout,
                        device=self.args.device
                        )
        self.dec = self.dec.to(self.args.device)    

        self.optim = torch.optim.Adam(
                        list(self.enc.parameters()) + \
                        list(self.dec.parameters()),
                        lr=self.args.lr,
                        betas=(self.args.beta1, 0.999)
                        )

    def save_model(self, path):
        print('Saving Model on %s ...' % (path))
        state = {
            'enc': self.enc.state_dict(),
            'dec': self.dec.state_dict()
        }
        torch.save(state, path)

    def load_model(self, path):
        print('Loading Model from %s ...' % (path))
        ckpt = torch.load(path, map_location=self.args.device)
        self.enc.load_state_dict(ckpt['enc'])
        self.dec.load_state_dict(ckpt['dec'])

    def zero_grad(self):
        self.enc.zero_grad()
        self.dec.zero_grad()

    def set_train(self):
        self.enc.train()
        self.dec.train()

    def set_eval(self):
        self.enc.eval()
        self.dec.eval()

    def train(self, data_loader):
        #with torch.autograd.set_detect_anomaly(True):
        self.epoch += 1
        self.set_train()
        record = utils.Record()
        record_acc = utils.Record()
        start_time = time.time()
        #progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start()
        progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
        for i, (trace, indexes, sent_length, name_list) in enumerate(data_loader):
            progress.update(i + 1)
            self.zero_grad()
            trace = trace.to(self.args.device)
            indexes = indexes.to(self.args.device)
            sent_length = sent_length.to(self.args.device)
            bs = trace.size(0)
    
            encoded = self.enc(trace)

            scores, caps_sorted, decode_lengths, alphas, sort_ind = self.dec(encoded, indexes, sent_length)
            targets = caps_sorted[:, 1:]
            
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            scores = scores.data
            targets = targets.data

            # Calculate loss
            loss = self.ce(scores, targets)

            # Add doubly stochastic attention regularization
            loss += self.args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            loss.backward()

            # Clip gradients
            if self.args.grad_clip is not None:
                utils.clip_gradient(self.optim, self.args.grad_clip)
            self.optim.step()

            record.add(loss.detach().item())

            topk = 1
            acc = utils.accuracy(scores.detach(), targets.detach(), topk)
            record_acc.add(float(acc))

            #torch.cuda.empty_cache()
            
        progress.finish()
        utils.clear_progressbar()
        print('----------------------------------------')
        print('Epoch: %d' % self.epoch)
        print('Costs Time: %.2f s' % (time.time() - start_time))
        print('Recons Loss: %f' % (record.mean()))
        print('Top %d Acc: %f' % (topk, record_acc.mean()))

    def test(self, data_loader):
        #with torch.autograd.set_detect_anomaly(True):
        self.set_train()
        record = utils.Record()
        record_acc = utils.Record()
        start_time = time.time()
        #progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start()
        progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
        with torch.no_grad():
            for i, (trace, indexes, sent_length, name_list) in enumerate(data_loader):
                progress.update(i + 1)

                trace = trace.to(self.args.device)
                indexes = indexes.to(self.args.device)
                sent_length = sent_length.to(self.args.device)
                bs = trace.size(0)
        
                encoded = self.enc(trace)

                scores, caps_sorted, decode_lengths, alphas, sort_ind = self.dec(encoded, indexes, sent_length)
                targets = caps_sorted[:, 1:]
                
                scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

                scores = scores.data
                targets = targets.data

                # Calculate loss
                loss = self.ce(scores, targets)

                # Add doubly stochastic attention regularization
                loss += self.args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

                record.add(loss.detach().item())

                topk = 1
                acc = utils.accuracy(scores.detach(), targets.detach(), topk)
                record_acc.add(float(acc))

            progress.finish()
            utils.clear_progressbar()
            print('----------------------------------------')
            print('Test.')
            print('Costs Time: %.2f s' % (time.time() - start_time))
            print('Recons Loss: %f' % (record.mean()))
            print('Top %d Acc: %f' % (topk, record_acc.mean()))

if __name__ == '__main__':
    import argparse
    import random

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import utils
    from params import Params
    from data_loader import DataLoader

    p = Params()
    args = p.parse()

    if args.cpu == 'amd':
        if args.cache == 'dcache':
            (args.trace_w, args.trace_c) = (256, 8)
        elif args.cache == 'icache':
            (args.trace_w, args.trace_c) = (256, 2)
    elif args.cpu == 'intel':
        if args.cache == 'dcache':
            (args.trace_w, args.trace_c) = (256, 4)
        elif args.cache == 'icache':
            (args.trace_w, args.trace_c) = (256, 4)

    args.nz = 512
    print(args.exp_name)

    manual_seed = random.randint(1, 10000)
    print('Manual Seed: %d' % manual_seed)

    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    utils.make_path(args.output_root)
    utils.make_path(args.output_root + args.exp_name)

    args.ckpt_root = args.output_root + args.exp_name + '/ckpt/'

    utils.make_path(args.ckpt_root)

    loader = DataLoader(args)

    train_dataset = RealSideDataset(args, split=args.data_path[args.dataset]['split'][0])
    test_dataset = RealSideDataset(args, split=args.data_path[args.dataset]['split'][1])
    args.vocab_size = len(train_dataset.word_dict.keys())

    engine = TextEngine(args, 
            idx2word=train_dataset.index_to_word,
            word2idx=train_dataset.word_to_index
        )

    train_loader = loader.get_loader(train_dataset)
    test_loader = loader.get_loader(test_dataset)

    for i in range(args.num_epoch):
        engine.train(train_loader)
        if i % args.test_freq == 0:
            engine.test(test_loader)
            engine.save_model((args.ckpt_root + '%03d.pth') % (i + 1))
    engine.save_model(args.ckpt_root + 'final.pth')
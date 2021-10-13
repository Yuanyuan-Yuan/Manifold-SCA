import time
import numpy as np
import progressbar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

import utils
import models

import gc

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
        self.enc = models.__dict__['trace_encoder_square_%d' % self.args.trace_w](dim=self.args.nz, nc=self.args.trace_c)
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
        progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start()
        # progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
        for i, (trace, indexes, sent_length, prefix) in enumerate(data_loader):
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
        progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start()
        # progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
        with torch.no_grad():
            for i, (trace, indexes, sent_length, prefix) in enumerate(data_loader):
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

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import utils
    from params import Params
    from data_loader import *
    # all datasets

    p = Params()
    args = p.parse()

    args.trace_c = 6
    args.trace_w = 128
    args.nz = 512

    print(args.exp_name)

    utils.make_path(args.output_root)
    utils.make_path(args.output_root + args.exp_name)

    args.ckpt_root = args.output_root + args.exp_name + '/ckpt/'

    utils.make_path(args.ckpt_root)

    loader = DataLoader(args)

    assert args.dataset == 'DailyDialog'
    args.pad_length = 64 # DailyDialog
    # args.pad_length = 181 # COCO
    train_dataset = DailyDialogDataset(
                    text_dir=args.data_path[args.dataset]['media'],
                    npz_dir=args.data_path[args.dataset][args.side],
                    dict_path=args.data_path[args.dataset]['vocab_path'],
                    split=args.data_path[args.dataset]['split'][0], 
                    side=args.side,
                    pad_length=args.pad_length,
                    trace_c=args.trace_c,
                    trace_w=args.trace_w
                )
    test_dataset = DailyDialogDataset(
                    text_dir=args.data_path[args.dataset]['media'],
                    npz_dir=args.data_path[args.dataset][args.side],
                    dict_path=args.data_path[args.dataset]['vocab_path'],
                    split=args.data_path[args.dataset]['split'][1], 
                    side=args.side,
                    pad_length=args.pad_length,
                    trace_c=args.trace_c,
                    trace_w=args.trace_w
                )
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
import time
import numpy as np
import progressbar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

import utils
import models

class Pipeline(object):
    def __init__(self, args):
        self.args = args
        self.epoch = 0
        self.mse = nn.MSELoss().to(self.args.device)
        self.l1 = nn.L1Loss().to(self.args.device)
        self.bce = nn.BCELoss().to(self.args.device)
        self.ce = nn.CrossEntropyLoss().to(self.args.device)
        self.real_label = 1
        self.fake_label = 0
        self.norm_inv = utils.NormalizeInverse((0.5,), (0.5,))
        self.init_model_optimizer()

    def init_model_optimizer(self):
        print('Initializing Model and Optimizer...')
        if self.args.media == 'image':
            self.enc = models.__dict__['attn_trace_encoder_256'](dim=128, nc=6)
            self.dec = models.__dict__['ResDecoder128'](dim=128, nc=3)
        elif self.args.media == 'audio':
            self.enc = models.__dict__['attn_trace_encoder_512'](dim=256, nc=8)
            self.dec = models.__dict__['audio_decoder_128'](dim=256, nc=1, out_s=44)
        elif self.args.media == 'text':
            self.enc = models.__dict__['trace_encoder_square_128'](dim=512, nc=6)
            self.dec = models.__dict__['DecoderWithAttention'](
                            attention_dim=self.args.attention_dim, 
                            embed_dim=self.args.embed_dim, 
                            decoder_dim=self.args.decoder_dim,
                            vocab_size=self.args.vocab_size,
                            encoder_dim=512,
                            dropout=self.args.dropout
                        )

        self.enc = self.enc.to(self.args.device)
        self.dec = self.dec.to(self.args.device)

    def load_model(self, path):
        print('Loading Model from %s ...' % (path))
        ckpt = torch.load(path, map_location=self.args.device)
        self.enc.load_state_dict(ckpt['enc'])
        self.dec.load_state_dict(ckpt['dec'])

    def save_grad(self, grad, name_list, path):
        #print('Saving Grad on %s ...' % (path))
        assert len(grad) == len(name_list)
        for i in range(len(grad)):
            np.savez_compressed(path + name_list[i] + '-grad.npz', grad[i])

    def set_train(self):
        self.enc.train()
        self.dec.train()

    def set_eval(self):
        self.enc.eval()
        self.dec.eval()

    def zero_grad(self):
        self.enc.zero_grad()
        self.dec.zero_grad()

    def process_grad(self, grad):
        bs = grad.size(0)
        grad = grad.view([bs, -1])
        for j in range(bs):
            grad[j] = abs(grad[j])
            grad[j] -= grad[j].min()
            grad[j] /= grad[j].max()
        grad = grad.detach().cpu().numpy()
        return grad

    def image_localize(self, data_loader):
        print('Image Grad.')
        self.set_train()
        start_time = time.time()
        progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start()
        for i, data in enumerate(data_loader):
            progress.update(i + 1)
            self.zero_grad()
            (trace, image, name_list, *_) = data
            trace = trace.to(self.args.device)
            image = image.to(self.args.device)
            grad_saver = utils.GradSaver()
            encoded = self.enc.forward_grad(trace, grad_saver)
            decoded = self.dec(encoded)

            recons_err = self.l1(decoded, image)
            recons_err.backward()

            gradient = self.process_grad(grad_saver.grad)

            self.save_grad(gradient, name_list, self.args.grad_root)
        progress.finish()
        print('----------------------------------------')
        print('Cost Time: %f' % (time.time() - start_time))
        print('Gradients Saved on %s' % self.args.grad_root)

    def audio_localize(self, data_loader):
        print('Audio Grad.')
        self.set_train()
        start_time = time.time()
        progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start()
        for i, data in enumerate(data_loader):
            progress.update(i + 1)
            self.zero_grad()
            (trace, audio, name_list, *_) = data
            trace = trace.to(self.args.device)
            audio = audio.to(self.args.device)
            grad_saver = utils.GradSaver()
            encoded = self.enc.forward_grad(trace, grad_saver)
            decoded = self.dec(encoded)

            recons_err = self.mse(decoded, audio)
            recons_err.backward()

            gradient = self.process_grad(grad_saver.grad)

            self.save_grad(gradient, name_list, self.args.grad_root)
        progress.finish()
        print('----------------------------------------')
        print('Cost Time: %f' % (time.time() - start_time))
        print('Gradients Saved on %s' % self.args.grad_root)

    def text_localize(self, data_loader):
        print('Text Grad.')
        self.set_train()
        start_time = time.time()
        progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start()
        for i, data in enumerate(data_loader):
            progress.update(i + 1)
            self.zero_grad()
            (trace, indexes, sent_length, name_list) = data
            trace = trace.to(self.args.device)
            indexes = indexes.to(self.args.device)
            sent_length = sent_length.to(self.args.device)
            grad_saver = utils.GradSaver()
            encoded = self.enc.forward_grad(trace, grad_saver)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = self.dec(encoded, indexes, sent_length)

            targets = caps_sorted[:, 1:]
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
            scores = scores.data
            targets = targets.data

            loss = self.ce(scores, targets)
            loss += self.args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            loss.backward()

            gradient = self.process_grad(grad_saver.grad)
            name_list = name_list[sort_ind]
            name_arr = [('%07d' % name.item()) for name in name_list]

            self.save_grad(gradient, name_arr, self.args.grad_root)
        progress.finish()
        print('----------------------------------------')
        print('Cost Time: %f' % (time.time() - start_time))
        print('Gradients Saved on %s' % self.args.grad_root)

    def pinpoint(self, grad_dir, inst_dir):
        grad_list = sorted(os.listdir(grad_dir))
        inst_list = sorted(os.listdir(inst_dir))

        result_dic = {}
        threshold_list = [0.4, 0.5, 0.6, 0.8, 0.9]
        for threshold in threshold_list:
            result_dic[threshold] = {}

        progress = progressbar.ProgressBar(maxval=len(inst_list), widgets=utils.get_widgets()).start()
        for i, inst_name in enumerate(inst_list):
            progress.update(i + 1)
            grad_name = inst_name.split('.')[0] + '-grad.npz'
            # inst_name = grad_name.split('-')[0] + '.out'
            grad = np.load(grad_dir + grad_name)['arr_0']
            with open(inst_dir + inst_name, 'r') as f:
                inst = f.readlines()
            for threshold in threshold_list:
                index = np.where((grad > threshold) == 1)[0]
                selected_addr = []
                for idx in index:
                    if idx >= len(inst):
                        continue
                    selected_inst = inst[idx]
                    [func_name, assembly, content] = selected_inst.strip().split('; ')
                    [ins_addr, op, mem_addr] = content.strip().split(' ')
                    selected_addr.append('%s %s %s' % (func_name, assembly, ins_addr))
                for addr in selected_addr:
                    if addr in result_dic[threshold].keys():
                        result_dic[threshold][addr] += 1
                    else:
                        result_dic[threshold][addr] = 1
        progress.finish()

        for threshold in threshold_list:
            dic_save = {k: result_dic[threshold][k] for k in sorted(result_dic[threshold], key=result_dic[threshold].get, reverse=True)}
            with open(('%s%s-threshold.json' % (self.args.localize_root, threshold)), 'w') as f:
                json.dump(dic_save, f, indent=2)


if __name__ == '__main__':
    import argparse
    import random

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import utils
    from params import Params
    from data_loader import *

    p = Params()
    args = p.parse()

    dataset2media = {
        'CelebA': 'image',
        'ChestX-ray': 'image',
        'SC09': 'audio',
        'Sub-URMP': 'audio',
        'COCO': 'text',
        'DailyDialog': 'text'
    }

    media2trace = {
        'image': (6, 256),
        'audio': (8, 512),
        'text': (6, 128)
    }

    media2nz = {
        'image': 128,
        'audio': 256,
        'text': 512
    }

    media = dataset2media[args.dataset]
    args.media = media
    (args.trace_c, args.trace_w) = media2trace[media]
    args.nz = media2nz[media]
    args.lms_w = 128
    if args.dataset == 'SC09':
        args.lms_h = 44
    elif args.dataset == 'Sub-URMP':
        args.lms_h = 22

    ROOT = os.environ.get('MANIFOLD_SCA')
    if ROOT is None:
        ROOT = '..'

    print(args.exp_name)

    manual_seed = random.randint(1, 10000)
    #manual_seed = 202
    print('Manual Seed: %d' % manual_seed)

    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    utils.make_path(args.output_root)
    utils.make_path(args.output_root + args.exp_name)

    args.ckpt_root = args.output_root + args.exp_name + '/ckpt/'
    args.grad_root = args.output_root + args.exp_name + '/grad/'
    args.localize_root = args.output_root + args.exp_name + '/localize/'

    utils.make_path(args.ckpt_root)
    utils.make_path(args.grad_root)
    utils.make_path(args.localize_root)

    loader = DataLoader(args)

    media_dataset = CelebaDataset(
                    img_dir=args.data_path[args.dataset]['media'], 
                    npz_dir=args.data_path[args.dataset][args.side],
                    ID_path=args.data_path[args.dataset]['ID_path'],
                    split=args.data_path[args.dataset]['split'][1],
                    trace_c=args.trace_c,
                    trace_w=args.trace_w,
                    image_size=args.image_size,
                    side=args.side
                )

    # media_dataset = SC09Dataset(
    #                 lms_dir=args.data_path[args.dataset]['media'],
    #                 npz_dir=args.data_path[args.dataset][args.side],
    #                 split=args.data_path[args.dataset]['split'][1],
    #                 trace_c=args.trace_c,
    #                 trace_w=args.trace_w,
    #                 max_db=args.max_db,
    #                 min_db=args.min_db,
    #                 side=args.side
    #             )

    # media_dataset = DailyDialogDataset(
    #                 text_dir=args.data_path[args.dataset]['media'],
    #                 npz_dir=args.data_path[args.dataset][args.side],
    #                 dict_path=args.data_path[args.dataset]['vocab_path'],
    #                 split=args.data_path[args.dataset]['split'][1], 
    #                 side=args.side,
    #                 pad_length=args.pad_length,
    #                 trace_c=args.trace_c,
    #                 trace_w=args.trace_w
    #             )

    
    media_loader = loader.get_loader(media_dataset, False)
    engine = Pipeline(args)

    model_path = ROOT + '/models/pin/CelebA_cacheline_pre/final.pth'
    # if you want to use your trained model,
    # comment the above line and uncomment the
    # following line.

    # model_path = (args.ckpt_root + 'final.pth')
    engine.load_model(model_path)

    engine.image_localize(media_loader)
    # engine.audio_localize(media_loader)
    # engine.text_localize(media_loader)

    engine.pinpoint(
            grad_dir=args.grad_root,
            inst_dir=args.data_path[args.dataset]['pin'] + 'test/'
        )

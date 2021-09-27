import time
import numpy as np
import progressbar
from nltk.translate.bleu_score import corpus_bleu

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
        self.mse = nn.MSELoss().cuda()
        self.l1 = nn.L1Loss().cuda()
        self.bce = nn.BCELoss().cuda()
        self.ce = nn.CrossEntropyLoss().cuda()
        self.real_label = 1
        self.fake_label = 0
        self.norm_inv = utils.NormalizeInverse((0.5,), (0.5,))
        self.init_model_optimizer()

    def init_model_optimizer(self):
        print('Initializing Model and Optimizer...')
        # self.enc = models.__dict__['attn_trace_encoder_256'](dim=self.args.image_nz, nc=6)
        # self.dec = models.__dict__['ResDecoder128'](dim=self.args.image_nz, nc=3)

        # self.enc = models.__dict__['attn_trace_encoder_512'](dim=self.args.audio_nz, nc=8)
        # self.dec = models.__dict__['audio_decoder_128'](dim=self.args.audio_nz, nc=1, out_s=22)

        self.enc = models.__dict__['trace_encoder_square_128'](dim=self.args.text_nz, nc=6)
        self.dec = models.__dict__['DecoderWithAttention'](
                        attention_dim=self.args.attention_dim, 
                        embed_dim=self.args.embed_dim, 
                        decoder_dim=self.args.decoder_dim,
                        vocab_size=self.args.vocab_size,
                        encoder_dim=self.args.text_nz,
                        dropout=self.args.dropout
                    )

        self.enc = torch.nn.DataParallel(self.enc).cuda()
        self.dec = torch.nn.DataParallel(self.dec).cuda()

    def load_model(self, path):
        print('Loading Model from %s ...' % (path))
        ckpt = torch.load(path)
        self.enc.module.load_state_dict(ckpt['enc'])
        self.dec.module.load_state_dict(ckpt['dec'])

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
        # print(grad[0])
        # print('Max: ', max(grad[0]))
        # print('Min: ', min(grad[0]))
        for j in range(bs):
            grad[j] = abs(grad[j])
            grad[j] -= grad[j].min()
            grad[j] /= grad[j].max()
        grad = grad.detach().cpu().numpy()
        # print('Shape: ', grad.shape)
        # print('Max: ', max(grad[0]))
        # print('Min: ', min(grad[0]))
        return grad

    def get_image_grad(self, data_loader):
        print('Image Grad.')
        self.set_train()
        start_time = time.time()
        progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
        for i, data in enumerate(data_loader):
            progress.update(i + 1)
            self.zero_grad()
            (trace, image, name_list, *_) = data
            trace = trace.cuda()
            image = image.cuda()
            grad_saver = utils.GradSaver()
            encoded = self.enc.module.forward_grad(trace, grad_saver)
            decoded = self.dec(encoded)

            recons_err = self.l1(decoded, image)
            recons_err.backward()

            gradient = self.process_grad(grad_saver.grad)

            self.save_grad(gradient, name_list, self.args.grad_path)
        progress.finish()
        print('----------------------------------------')
        print('Cost Time: %f' % (time.time() - start_time))
        print('Saved Gradients on %s' % self.args.grad_path)

    def get_audio_grad(self, data_loader):
        print('Audio Grad.')
        self.set_train()
        start_time = time.time()
        progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
        for i, data in enumerate(data_loader):
            progress.update(i + 1)
            self.zero_grad()
            (trace, audio, name_list, *_) = data
            trace = trace.cuda()
            audio = audio.cuda()
            grad_saver = utils.GradSaver()
            encoded = self.enc.module.forward_grad(trace, grad_saver)
            decoded = self.dec(encoded)

            recons_err = self.mse(decoded, audio)
            recons_err.backward()

            gradient = self.process_grad(grad_saver.grad)

            self.save_grad(gradient, name_list, self.args.grad_path)
        progress.finish()
        print('----------------------------------------')
        print('Cost Time: %f' % (time.time() - start_time))
        print('Saved Gradients on %s' % self.args.grad_path)

    def get_text_grad(self, data_loader):
        print('Text Grad.')
        self.set_train()
        start_time = time.time()
        progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
        for i, data in enumerate(data_loader):
            progress.update(i + 1)
            self.zero_grad()
            (trace, indexes, sent_length, name_list) = data
            trace = trace.cuda()
            indexes = indexes.cuda()
            sent_length = sent_length.cuda()
            grad_saver = utils.GradSaver()
            encoded = self.enc.module.forward_grad(trace, grad_saver)
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

            self.save_grad(gradient, name_arr, self.args.grad_path)
        progress.finish()
        print('----------------------------------------')
        print('Cost Time: %f' % (time.time() - start_time))
        print('Saved Gradients on %s' % self.args.grad_path)

if __name__ == '__main__':
    import argparse
    import random

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import utils
    from data_loader import *

    name_list = ['celeba', 'chestX-ray8', 'SC09', 'Sub-URMP', 'COCO_caption', 'ACL_abstract', 'DailyDialog']
    
    side = 'cacheline'
    dataset_name = 'DailyDialog'
    assert dataset_name in name_list

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default=('%s_%s' % (dataset_name, side)))
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--output_root', type=str, default='/root/output/gradient_test/')

    parser.add_argument('--image_nz', type=int, default=128)
    parser.add_argument('--audio_nz', type=int, default=256)
    parser.add_argument('--text_nz', type=int, default=512)

    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--attention_dim', type=int, default=512)
    parser.add_argument('--decoder_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--grad_clip', type=float, default=5)
    parser.add_argument('--alpha_c', type=float, default=1)
    parser.add_argument('--vocab_size', type=int, default=-1)
    parser.add_argument('--pad_length', type=int, default=-1)
    parser.add_argument('--max_db', type=float, default=45)
    parser.add_argument('--min_db', type=float, default=-100)

    args = parser.parse_args()

    print(args.exp_name)

    manual_seed = random.randint(1, 10000)
    #manual_seed = 202
    print('Manual Seed: %d' % manual_seed)

    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    utils.make_path(args.output_root)
    utils.make_path(args.output_root + args.exp_name)

    args.grad_path = args.output_root + args.exp_name + '/'

    loader = DataLoader(args)

    # TEXT START
    # media_dataset, args.pad_length = ACLDataset(
    #                 data_json_path='/root/data/ACL_text/acl_abstract_test.json',
    #                 npz_dir=('/root/data/ACL_%s/' % side), 
    #                 split='test', 
    #                 dict_path='/root/data/ACL_text/ACL_word_dict_freq5.json',
    #                 pad_length=250,
    #                 side=side
    #             ), 250

    # media_dataset, args.pad_length = CaptionDataset(
    #                 data_json_path='/root/data/COCO_caption_text/val_sentences.json',
    #                 npz_dir=('/root/data/COCO_caption_%s/' % side), 
    #                 split='test', 
    #                 dict_path='/root/data/COCO_caption_text/word_dict_freq5.json',
    #                 pad_length=181,
    #                 side=side
    #             ), 181

    media_dataset, args.pad_length = DailyDialogDataset(
                    data_json_path='/root/data/DailyDialog_text/dialogues_test.json',
                    npz_dir=('/root/data/DailyDialog_%s/' % side), 
                    split='test', 
                    dict_path='/root/data/DailyDialog_text/DailyDialog_word_dict_freq5.json',
                    pad_length=64,
                    side=side
                ), 64

    args.vocab_size = len(media_dataset.word_dict.keys())
    # TEXT END

    # IMAGE START
    # media_dataset = ChestDataset(
    #                 img_dir='/root/data/ChestX-ray8_jpg128/', 
    #                 npz_dir=('/root/data/ChestX-ray8_%s/' % side), 
    #                 split='test',
    #                 image_size=128,
    #                 side=side
    #             )

    # media_dataset = CelebaDataset(
    #                 img_dir='/root/data/celeba_crop128/', 
    #                 npz_dir=('/root/data/celeba_crop128_%s/' % side), 
    #                 split='test',
    #                 image_size=128,
    #                 side=side
    #             )
    # IMAGE END

    # AUDIO START
    # media_dataset = SC09Dataset(
    #                 lms_dir='/root/data/sc09_raw_lms/',
    #                 npz_dir=('/root/data/sc09_ar2000_%s_%s/' % ('upsample', side)),
    #                 split='test',
    #                 max_db=45,
    #                 min_db=-100,
    #                 side=side
    #             )
    # media_dataset = URMPDataset(
    #                 lms_dir='/root/data/Sub-URMP_raw_lms/',
    #                 npz_dir=('/root/data/Sub-URMP_ar2000_%s_%s/' % ('upsample', side)),
    #                 split='test',
    #                 max_db=45,
    #                 min_db=-100,
    #                 side=side
    #             )
    # AUDIO END
    
    media_loader = loader.get_loader(media_dataset, False)
    model = Pipeline(args)
    
    model_root = '/root/output/SCA/'
    # model_path = model_root + ('final_pretrain_%s_%s/ckpt/%03d.pth' % (dataset_name, side, 196))
    # model_path = model_root + ('final_gen_%s_%s_%s/ckpt/096.pth' % (dataset_name, 'upsample', side))
    model_path = model_root + ('final_%s_%s/ckpt/final.pth' % (dataset_name, side))

    model.load_model(model_path)

    # model.get_image_grad(media_loader)
    # model.get_audio_grad(media_loader)
    model.get_text_grad(media_loader)

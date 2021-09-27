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

class Text(object):
    def __init__(self, args, idx2word, word2idx):
        self.args = args
        self.epoch = 0
        self.mse = nn.MSELoss().cuda()
        self.l1 = nn.L1Loss().cuda()
        self.bce = nn.BCELoss().cuda()
        self.ce = nn.CrossEntropyLoss().cuda()
        self.last_output = None
        self.idx2word = idx2word # function
        self.word2idx = word2idx
        self.init_model_optimizer()

    def init_model_optimizer(self):
        print('Initializing Model and Optimizer...')
        self.enc = models.__dict__['trace_encoder_square_128'](dim=self.args.nz, nc=6)
        self.enc = torch.nn.DataParallel(self.enc).cuda()

        self.dec = models.__dict__['DecoderWithAttention'](
                        attention_dim=self.args.attention_dim, 
                        embed_dim=self.args.embed_dim, 
                        decoder_dim=self.args.decoder_dim,
                        vocab_size=self.args.vocab_size,
                        encoder_dim=self.args.nz,
                        dropout=self.args.dropout
                        )
        self.dec = torch.nn.DataParallel(self.dec).cuda()    

        self.optim = torch.optim.Adam(
                        list(self.enc.module.parameters()) + \
                        list(self.dec.module.parameters()),
                        lr=self.args.lr,
                        betas=(self.args.beta1, 0.999)
                        )

    def save_model(self, path):
        print('Saving Model on %s ...' % (path))
        state = {
            'enc': self.enc.module.state_dict(),
            'dec': self.dec.module.state_dict()
        }
        torch.save(state, path)

    def load_model(self, path):
        print('Loading Model from %s ...' % (path))
        ckpt = torch.load(path)
        self.enc.module.load_state_dict(ckpt['enc'])
        self.dec.module.load_state_dict(ckpt['dec'])

    def save_output(self, beam_size, encoded, indexes, path):
        with torch.no_grad():
            bs = encoded.size(0)
            with open(path, 'w') as f:
                f.write('BEAM SIZE: %d\n' % beam_size)
                f.write('%s\t%s\n' % ('RECOUNSTRUCTED', 'TARGET'))
                for i in range(bs):
                    encoder_out = encoded[i]
                    target = indexes[i]
                    k = beam_size
                    encoder_out = encoder_out.view(1, -1, self.args.nz)
                    num_pixels = encoder_out.size(1)
                    encoder_out = encoder_out.expand(k, num_pixels, self.args.nz)
                    k_prev_words = torch.LongTensor([[self.word2idx('<START>')]] * k).cuda() # (k, 1)
                    seqs = k_prev_words  # (k, 1)
                    top_k_scores = torch.zeros(k, 1).cuda()
                    complete_seqs = []
                    complete_seqs_scores = []

                    decoder = self.dec.module

                    step = 1
                    h, c = decoder.init_hidden_state(encoder_out)

                    # s <= k
                    while True:
                        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
                        awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

                        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
                        awe = gate * awe

                        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

                        scores = decoder.fc(h)  # (s, vocab_size)
                        scores = F.log_softmax(scores, dim=1)

                        # Add
                        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

                        # For the first step, all k points will have the same scores (since same k previous words, h, c)
                        if step == 1:
                            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                        else:
                            # Unroll and find top scores, and their unrolled indices
                            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                        # Convert unrolled indices to actual indices of scores
                        prev_word_inds = top_k_words / self.args.vocab_size  # (s)
                        next_word_inds = top_k_words % self.args.vocab_size  # (s)

                        # Add new words to sequences
                        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

                        # Which sequences are incomplete (didn't reach <end>)?
                        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                           next_word != self.word2idx('<END>')]
                        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                        # Set aside complete sequences
                        if len(complete_inds) > 0:
                            complete_seqs.extend(seqs[complete_inds].tolist())
                            complete_seqs_scores.extend(top_k_scores[complete_inds])
                        k -= len(complete_inds)  # reduce beam length accordingly

                        # Proceed with incomplete sequences
                        if k == 0:
                            break
                        seqs = seqs[incomplete_inds]
                        h = h[prev_word_inds[incomplete_inds]]
                        c = c[prev_word_inds[incomplete_inds]]
                        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

                        # Break if things have been going on too long
                        if step > 50:
                            break
                        step += 1

                    i = complete_seqs_scores.index(max(complete_seqs_scores))
                    seq = complete_seqs[i]
                    
                    seq_filtered = [
                        idx for idx in seq if self.idx2word(idx) not in ['<START>', '<END>', '<PAD>', '<UNK>']
                    ]

                    target_filtered = [
                        idx for idx in target if self.idx2word(int(idx.item())) not in ['<START>', '<END>', '<PAD>', '<UNK>']
                    ]

                    word_seq = ' '.join([
                        self.idx2word(idx) for idx in seq_filtered
                    ])

                    target_seq = ' '.join([
                        self.idx2word(int(idx.item())) for idx in target_filtered
                    ])

                    f.write('%s\t%s\n' % (word_seq, target_seq))
                    # Calculate BLEU-4 scores
                    # bleu4 = corpus_bleu([[target_filtered]], [seq_filtered])
                    # f.write('BLEU-4 Score: %f\n' % bleu4)

    def zero_grad(self):
        self.enc.zero_grad()
        self.dec.zero_grad()

    def set_train(self):
        self.enc.train()
        self.dec.train()

    def set_eval(self):
        #self.enc.eval()
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
        for i, (trace, indexes, sent_length) in enumerate(data_loader):
            progress.update(i + 1)
            self.zero_grad()
            trace = trace.cuda()
            indexes = indexes.cuda()
            sent_length = sent_length.cuda()
            bs = trace.size(0)
            # print('trace: ', trace.shape)
            # print('indexes: ', indexes.shape)
            # print('sent_length: ', sent_length.shape)
    
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
        # self.save_output(
        #             beam_size=1, 
        #             encoded=encoded.detach(), 
        #             indexes=indexes, 
        #             path=((self.args.text_root+'train_%03d.txt') % self.epoch)
        #         )
        #self.save_output(output_seq, ((self.args.code_root+'train_%03d.txt') % self.epoch))
        # fake = self.G(self.fixed_noise)
        #utils.save_image(decoded.data, (self.args.image_root+'train_%03d.jpg') % self.epoch)

    def test(self, data_loader):
        #with torch.autograd.set_detect_anomaly(True):
        self.set_train()
        record = utils.Record()
        record_acc = utils.Record()
        start_time = time.time()
        #progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start()
        progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
        with torch.no_grad():
            for i, (trace, indexes, sent_length) in enumerate(data_loader):
                progress.update(i + 1)

                trace = trace.cuda()
                indexes = indexes.cuda()
                sent_length = sent_length.cuda()
                bs = trace.size(0)
                # print('trace: ', trace.shape)
                # print('indexes: ', indexes.shape)
                # print('sent_length: ', sent_length.shape)
        
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

            #self.last_output = output_seq
            progress.finish()
            utils.clear_progressbar()
            print('----------------------------------------')
            print('Test.')
            print('Costs Time: %.2f s' % (time.time() - start_time))
            print('Recons Loss: %f' % (record.mean()))
            print('Top %d Acc: %f' % (topk, record_acc.mean()))
            beam_size = 1
            # self.save_output(
            #         beam_size=beam_size, 
            #         encoded=encoded, 
            #         indexes=indexes, 
            #         path=((self.args.text_root+'test_%03d_beamsize_%d.txt') % (self.epoch, beam_size))
            #     )
            #utils.save_image(decoded.data, (self.args.image_root + 'test_%03d.jpg') % self.epoch)
            #utils.save_image(image.data, (self.args.image_root + 'target_%03d.jpg') % self.epoch)


if __name__ == '__main__':
    import argparse

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import utils
    from data_loader import *

    parser = argparse.ArgumentParser()

    side = 'cacheline'

    parser.add_argument('--exp_name', type=str, default=('final_DailyDialog_%s' % side))

    # parser.add_argument('--npz_dir', type=str, default=('/root/data/COCO_caption_%s/' % side))
    # parser.add_argument('--vocab_path', type=str, default='/root/data/COCO_caption_text/word_dict_freq5.json')
    parser.add_argument('--npz_dir', type=str, default=('/root/data/DailyDialog_%s/' % side))
    parser.add_argument('--vocab_path', type=str, default='/root/data/DailyDialog_text/DailyDialog_word_dict_freq5.json')
    parser.add_argument('--output_root', type=str, default='/root/output/SCA/')

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--nz', type=int, default=512)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--test_freq', type=int, default=5)
    parser.add_argument('--train_in_step', type=int, default=30)
    parser.add_argument('--test_in_step', type=int, default=30)

    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--attention_dim', type=int, default=512)
    parser.add_argument('--decoder_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--grad_clip', type=float, default=5)
    parser.add_argument('--alpha_c', type=float, default=1)
    parser.add_argument('--vocab_size', type=int, default=-1)

    args = parser.parse_args()

    print(args.exp_name)

    utils.make_path(args.output_root)
    utils.make_path(args.output_root + args.exp_name)

    args.text_root = args.output_root + args.exp_name + '/text/'
    args.ckpt_root = args.output_root + args.exp_name + '/ckpt/'

    utils.make_path(args.text_root)
    utils.make_path(args.ckpt_root)

    with open(args.output_root + args.exp_name + '/args.json', 'w') as f:
        json.dump(args.__dict__, f)

    loader = DataLoader(args)

    # train_dataset = CaptionDataset(
    #                 data_json_path='/root/data/COCO_caption_text/train_sentences.json', 
    #                 npz_dir=args.npz_dir, 
    #                 split='train', 
    #                 dict_path=args.vocab_path,
    #                 pad_length=181,
    #                 side=side
    #             )
    # test_dataset = CaptionDataset(
    #                 data_json_path='/root/data/COCO_caption_text/val_sentences.json',
    #                 npz_dir=args.npz_dir, 
    #                 split='test', 
    #                 dict_path=args.vocab_path,
    #                 pad_length=181,
    #                 side=side
    #             )

    train_dataset = DailyDialogDataset(
                    data_json_path='/root/data/DailyDialog_text/dialogues_train.json', 
                    npz_dir=args.npz_dir, 
                    split='train', 
                    dict_path=args.vocab_path,
                    pad_length=64,
                    side=side
                )
    test_dataset = DailyDialogDataset(
                    data_json_path='/root/data/DailyDialog_text/dialogues_test.json',
                    npz_dir=args.npz_dir, 
                    split='test', 
                    dict_path=args.vocab_path,
                    pad_length=64,
                    side=side
                )

    args.vocab_size = len(train_dataset.word_dict.keys())

    model = Text(args, 
            idx2word=train_dataset.index_to_word,
            word2idx=train_dataset.word_to_index
        )

    #model.load_model(args.ckpt_root + '%03d.pth' % 26)

    train_loader = loader.get_loader(train_dataset)
    test_loader = loader.get_loader(test_dataset)

    #model.test(test_loader)

    for i in range(args.num_epoch):
        model.train(train_loader)
        if (i + 0) % args.test_freq == 0:
            model.test(test_loader)
            model.save_model((args.ckpt_root + '%03d.pth') % (i + 1))

    model.save_model(args.ckpt_root + 'final.pth')

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
    def __init__(self, args, media, idx2word, word2idx):
        self.args = args
        self.epoch = 0
        self.mse = nn.MSELoss().cuda()
        self.l1 = nn.L1Loss().cuda()
        self.bce = nn.BCELoss().cuda()
        self.ce = nn.CrossEntropyLoss().cuda()
        self.real_label = 1
        self.fake_label = 0
        self.idx2word = idx2word # function
        self.word2idx = word2idx # function
        self.norm_inv = utils.NormalizeInverse((0.5,), (0.5,))
        self.init_model_optimizer()

    def init_model_optimizer(self):
        print('Initializing Model and Optimizer...')
        assert media in ['image', 'audio', 'text']
        if media == 'image':
            self.image_enc = models.__dict__['attn_trace_encoder_256'](dim=self.args.image_nz, nc=6)
            self.image_enc = torch.nn.DataParallel(self.image_enc).cuda()
            self.image_dec = models.__dict__['image_decoder_128'](dim=self.args.image_nz, nc=3)
            self.image_dec = torch.nn.DataParallel(self.image_dec).cuda()
        elif media == 'audio':
            self.audio_enc = models.__dict__['attn_trace_encoder_512'](dim=self.args.audio_nz, nc=8)
            self.audio_enc = torch.nn.DataParallel(self.audio_enc).cuda()
            self.audio_dec = models.__dict__['audio_decoder_128'](dim=self.args.audio_nz, nc=1, out_s=22)
            self.audio_dec = torch.nn.DataParallel(self.audio_dec).cuda()
        else:
            self.text_enc = models.__dict__['trace_encoder_square_128'](dim=self.args.text_nz, nc=6)
            self.text_enc = torch.nn.DataParallel(self.text_enc).cuda()
            self.text_dec = models.__dict__['DecoderWithAttention'](
                                attention_dim=self.args.attention_dim, 
                                embed_dim=self.args.embed_dim, 
                                decoder_dim=self.args.decoder_dim,
                                vocab_size=self.args.vocab_size,
                                encoder_dim=self.args.text_nz,
                                dropout=self.args.dropout
                            )
            self.text_dec = torch.nn.DataParallel(self.text_dec).cuda()

    def load_audio_cls(self, path):
        print('Loading Audio Classifier from %s ...' % path)
        ckpt = torch.load(path)
        self.E = models.__dict__['audio_output_embed_128'](dim=self.args.audio_nz, nc=1)
        self.E = torch.nn.DataParallel(self.E).cuda()
        self.C2 = models.__dict__['classifier'](dim=self.args.audio_nz, n_class=10, use_bn=True)
        self.C2 = torch.nn.DataParallel(self.C2).cuda()
        self.E.module.load_state_dict(ckpt['E'])
        self.C2.module.load_state_dict(ckpt['C2'])

    def load_image_model(self, path):
        print('Loading Image Model from %s ...' % (path))
        ckpt = torch.load(path)
        self.image_enc.module.load_state_dict(ckpt['enc'])
        self.image_dec.module.load_state_dict(ckpt['dec'])

    def load_audio_model(self, path):
        print('Loading Audio Model from %s ...' % (path))
        ckpt = torch.load(path)
        self.audio_enc.module.load_state_dict(ckpt['enc'])
        self.audio_dec.module.load_state_dict(ckpt['dec'])

    def load_text_model(self, path):
        print('Loading Text Model from %s ...' % (path))
        ckpt = torch.load(path)
        self.text_enc.module.load_state_dict(ckpt['enc'])
        self.text_dec.module.load_state_dict(ckpt['dec'])

    def load_refiner(self, path):
        print('Loading Refiner from %s ...' % (path))
        ckpt = torch.load(path)
        self.refiner.module.load_state_dict(ckpt['G'])

    # def save_image(self, output, path):
    #     utils.save_image(output.data, path, nrow=10, normalize=True)

    def save_image(self, output, name_list, path):
        assert len(output) == len(name_list)
        for i in range(len(output)):
            utils.save_image(output[i].unsqueeze(0).data,
                             path + name_list[i] + '.jpg',
                             normalize=True, nrow=1, padding=0)

    def save_audio(self, output, name_list, path):
        # save lms in .npz format
        assert len(output) == len(name_list)
        for i in range(len(output)):
            output[i] = self.norm_inv(output[i])
        output = utils.my_scale_inv(v=output, 
                                        v_max=self.args.max_db,
                                        v_min=self.args.min_db)
        output = output.cpu().numpy()
        for i in range(len(output)):
            np.savez_compressed(path + name_list[i] + '.npz',
                                output[i])

    def save_text(self, beam_size, encoded, indexes, sent_length, path, teacher_force=True, pad_length=250): # COCO: 181, ACL: 250
        with torch.no_grad():
            bs = encoded.size(0)
            #record = utils.Record()
            with open(path, 'w') as f:
                f.write('TEACHER FORCE: %d\n' % int(teacher_force))
                f.write('BEAM SIZE: %d\n' % beam_size)
                f.write('%s\t%s\n' % ('RECOUNSTRUCTED', 'TARGET'))
                if teacher_force:
                    scores, caps_sorted, decode_lengths, *_ = self.text_dec(encoded, indexes, sent_length)
                    targets = caps_sorted[:, 1:]
                
                    scores_pack = pack_padded_sequence(scores, decode_lengths, batch_first=True)
                    targets_pack = pack_padded_sequence(targets, decode_lengths, batch_first=True)

                    scores_pack = scores_pack.data
                    targets_pack = targets_pack.data
                    topk = 1
                    acc = utils.accuracy(scores_pack.detach(), targets_pack.detach(), topk)
                    #print('ACCURACY: %f' % acc)
                    for i in range(bs):
                        prediction = scores[i] # [seq_len, vocab_size]
                        index_list = targets[i]
                        # print('prediction: ', prediction.shape)
                        # print('index_list: ', index_list.shape)
                        # acc = utils.accuracy(prediction.detach(), index_list.detach(), topk)
                        # print('Acc: %f' % acc)
                        val_seq, idx_seq = prediction.max(-1)
                        pred_sentence = [self.idx2word(int(idx.item())) for idx in idx_seq]
                        #pred_filtered = pred_sentence
                        pred_filtered = [word for word in pred_sentence if word not in ['<START>', '<END>', '<PAD>', '<UNK>']]
                        pred_seq = ' '.join(pred_filtered)

                        target_sentence = [self.idx2word(int(idx.item())) for idx in index_list]
                        #target_filtered = target_sentence
                        target_filtered = [word for word in target_sentence if word not in ['<START>', '<END>', '<PAD>', '<UNK>']]
                        target_seq = ' '.join(target_filtered)

                        f.write('%s\t%s\n' % (pred_seq, target_seq))
                        bleu4 = corpus_bleu([[target_filtered]], [pred_filtered])
                        #record.add(bleu4)
                        f.write('BLEU-4 Score: %f\n' % bleu4)
                else:
                    for i in range(bs):
                        encoder_out = encoded[i]
                        target = indexes[i]
                        k = beam_size
                        encoder_out = encoder_out.view(1, -1, self.args.text_nz)
                        num_pixels = encoder_out.size(1)
                        encoder_out = encoder_out.expand(k, num_pixels, self.args.text_nz)
                        k_prev_words = torch.LongTensor([[self.word2idx('<START>')]] * k).cuda() # (k, 1)
                        seqs = k_prev_words  # (k, 1)
                        top_k_scores = torch.zeros(k, 1).cuda()
                        complete_seqs = []
                        complete_seqs_scores = []
                        incomplete_seqs = []
                        incomplete_seqs_scores = []
                        
                        decoder = self.text_dec.module

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

                            incomplete_seqs.extend(seqs.tolist())
                            incomplete_seqs_scores.extend(top_k_scores)

                            # Break if things have been going on too long
                            if step > pad_length:
                                break
                            step += 1
                        if len(complete_seqs_scores):
                            i = complete_seqs_scores.index(max(complete_seqs_scores))
                            seq = complete_seqs[i]
                        else:
                            i = incomplete_seqs_scores.index(max(incomplete_seqs_scores))
                            seq = incomplete_seqs[i]

                        # seq_filtered = [
                        #     idx for idx in seq if self.idx2word(idx) not in ['<START>', '<END>', '<PAD>', '<UNK>']
                        # ]
                        seq_filtered = seq

                        # target_filtered = [
                        #     idx for idx in target if self.idx2word(int(idx.item())) not in ['<START>', '<END>', '<PAD>', '<UNK>']
                        # ]
                        target_filtered = target

                        word_seq = ' '.join([
                            self.idx2word(idx) for idx in seq_filtered
                        ])

                        target_seq = ' '.join([
                            self.idx2word(int(idx.item())) for idx in target_filtered
                        ])

                        f.write('%s\t%s\n' % (word_seq, target_seq))
                        bleu4 = corpus_bleu([[target_filtered]], [seq_filtered])
                        #record.add(bleu4)
                        f.write('BLEU-4 Score: %f\n' % bleu4)
                #print('BLEU-4 Score: %f' % record.mean())

    def set_train(self):
        self.image_enc.train()
        self.image_dec.train()
        self.audio_enc.train()
        self.audio_dec.train()
        self.text_enc.train()
        self.text_dec.train()
        self.refiner.train()

    def set_eval(self):
        self.image_enc.eval()
        self.image_dec.eval()
        self.audio_enc.eval()
        self.audio_dec.eval()
        self.text_enc.eval()
        self.text_dec.eval()
        self.refiner.eval()

    def evaluation(self, data_loader, media):
        assert media in ['image', 'audio', 'text']
        with torch.no_grad():
        #self.set_eval()
            if media == 'image':
                (encoder, decoder) = self.image_enc, self.image_dec
                metric = self.l1
            elif media == 'audio':
                (encoder, decoder) = self.audio_enc, self.audio_dec
                metric = self.mse
            else:
                (encoder, decoder) = self.text_enc, self.text_dec
                metric = self.ce

            record = utils.Record()
            record_acc = utils.Record()
            start_time = time.time()
            progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
            for i, data in enumerate(data_loader):
                progress.update(i + 1)
                if media == 'text':
                    (trace, indexes, sent_length, *_) = data
                    trace = trace.cuda()
                    indexes = indexes.cuda()
                    sent_length = sent_length.cuda()
                    
                    encoded = encoder(trace)
                    # encoded = encoder(torch.randn(trace.size()).cuda())
                    encoded = torch.randn(encoded.size()).cuda()

                    scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(encoded, indexes, sent_length)
                    targets = caps_sorted[:, 1:]
                    
                    scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
                    targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

                    scores = scores.data
                    targets = targets.data

                    loss = metric(scores, targets)
                    record.add(loss.item())
                    topk = 1
                    acc = utils.accuracy(scores.detach(), targets.detach(), topk)
                    record_acc.add(float(acc))
                elif media == 'audio':
                    (trace, target_media, name_list, content, ID) = data
                    target_media = target_media.cuda()
                    trace = trace.cuda()
                    content = content.cuda()
                    ID = ID.cuda()
                    encoded = encoder(trace)
                    # encoded = encoder(torch.randn(trace.size()).cuda())
                    encoded = torch.randn(encoded.size()).cuda()
                    decoded = decoder(encoded)
                    embed = self.E(decoded)
                    pred_content = self.C2(embed)
                    topk = 1
                    acc = utils.accuracy(pred_content.detach(), content, topk)
                    record_acc.add(float(acc))
                    loss = metric(decoded, target_media)
                    record.add(loss.item())
                else:
                    (trace, target_media, *_) = data
                    trace = trace.cuda()
                    target_media = target_media.cuda()
                    encoded = encoder(trace)
                    # encoded = encoder(torch.randn(trace.size()).cuda())
                    encoded = torch.randn(encoded.size()).cuda()
                    decoded = decoder(encoded)

                    refined = self.refiner(decoded)
                    loss = metric(refined, target_media)
                    #loss = metric(decoded, target_media)
                    record.add(loss.item())
            progress.finish()
            print('------------------%s------------------' % (media.capitalize()))
            print('Cost Time: %f' % (time.time() - start_time))
            print('Loss: %f' % record.mean())
            if media in ['text', 'audio']:
                print('Top %d Acc: %f' % (topk, record_acc.mean()))
            print('----------------------------------------')

    def output(self, data_loader, media, recons_path, target_path):
        assert media in ['image', 'audio', 'text']
        with torch.no_grad():
            #self.set_eval()
            if media == 'image':
                (encoder, decoder) = self.image_enc, self.image_dec
                #save_media = self.save_image
                suffix = 'jpg'
            elif media == 'audio':
                (encoder, decoder) = self.audio_enc, self.audio_dec
                #save_media = self.save_audio
                suffix = 'npz'
            else:
                (encoder, decoder) = self.text_enc, self.text_dec
                #save_media = self.save_text
                suffix = 'txt'
            start_time = time.time()
            progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
            for i, data in enumerate(data_loader):
                progress.update(i + 1)
                if media == 'text':
                    (trace, indexes, sent_length, *_) = data
                    trace = trace.cuda()
                    indexes = indexes.cuda()
                    sent_length = sent_length.cuda()
                    encoded = encoder(trace)
                    # encoded = encoder(torch.randn(trace.size()).cuda())
                    encoded = torch.randn(encoded.size()).cuda()
                    beam_size = 3
                    self.save_text(beam_size=3,
                                   encoded=encoded,
                                   indexes=indexes,
                                   sent_length=sent_length,
                                   path=(recons_path+'%s_%06d.%s') % (media, i, suffix),
                                   teacher_force=True,
                                   pad_length=self.args.pad_length)
                elif media == 'image':
                    (trace, target_media, name_list, *_) = data
                    trace = trace.cuda()
                    target_media = target_media.cuda()
                    encoded = encoder(trace)
                    # encoded = encoder(torch.randn(trace.size()).cuda())
                    encoded = torch.randn(encoded.size()).cuda()
                    decoded = decoder(encoded)
                    refined = self.refiner(decoded)
                    # self.save_image(refined.detach(), (recons_path+'%06d.%s') % (i, suffix))
                    # self.save_image(target_media, (target_path+'%06d.%s') % (i, suffix))
                    self.save_image(refined.detach(), name_list, recons_path)
                    self.save_image(target_media, name_list, target_path)
                else:
                    (trace, target_media, name_list, *_) = data
                    trace = trace.cuda()
                    target_media = target_media.cuda()
                    encoded = encoder(trace)
                    # encoded = encoder(torch.randn(trace.size()).cuda())
                    encoded = torch.randn(encoded.size()).cuda()
                    decoded = decoder(encoded)
                    self.save_audio(decoded.detach(), name_list, recons_path)
                    self.save_audio(target_media, name_list, target_path)

                # else:
                #     (trace, target_media, *_) = data
                #     trace = trace.cuda()
                #     target_media = target_media.cuda()
                #     encoded = encoder(trace)
                #     decoded = decoder(encoded)
                #     if media == 'image':
                #         refined = self.refiner(decoded)
                #         #refined = decoded
                #         save_media(refined.detach(), (recons_path+'%06d.%s') % (i, suffix))
                #     else:
                #         save_media(decoded.detach(), (recons_path+'%06d.%s') % (i, suffix))
                    
                #     save_media(target_media, (target_path+'%06d.%s') % (i, suffix))
            progress.finish()
            print('------------------%s------------------' % (media.capitalize()))
            print('Cost Time: %f' % (time.time() - start_time))
            print('Output Saved.')
            print('----------------------------------------')

if __name__ == '__main__':
    import argparse
    import random

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import utils
    from data_loader import *

    name_list = ['celeba', 'chestX-ray8', 'SC09', 'Sub-URMP', 'COCO_caption', 'ACL_abstract', 'DailyDialog']
    
    side = 'pagetable'
    dataset_name = 'chestX-ray8'
    assert dataset_name in name_list
    media = 'image'

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default=('%s_%s' % (dataset_name, side)))
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--output_root', type=str, default='/root/output/media_baseline/')

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

    args.recons_root = args.output_root + args.exp_name + '/recons/'
    args.target_root = args.output_root + args.exp_name + '/target/'
    utils.make_path(args.recons_root)
    utils.make_path(args.target_root)

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

    # media_dataset, args.pad_length = DailyDialogDataset(
    #                 data_json_path='/root/data/DailyDialog_text/dialogues_test.json',
    #                 npz_dir=('/root/data/DailyDialog_%s/' % side), 
    #                 split='test', 
    #                 dict_path='/root/data/DailyDialog_text/DailyDialog_word_dict_freq5.json',
    #                 pad_length=64,
    #                 side=side
    #             ), 64

    # args.vocab_size = len(media_dataset.word_dict.keys())
    # TEXT END

    # IMAGE START
    media_dataset = ChestDataset(
                    img_dir='/root/data/ChestX-ray8_jpg128/', 
                    npz_dir=('/root/data/ChestX-ray8_%s/' % side), 
                    split='test',
                    image_size=128,
                    side=side
                )

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
    model = Pipeline(args,
                media=media,
                idx2word=(media_dataset.index_to_word if media == 'text' else None),
                word2idx=(media_dataset.word_to_index if media == 'text' else None)
            )
    
    model_root = '/root/output/SCA/'

    # image_model_path = model_root + ('final_gen_%s_%s/ckpt/%03d.pth' % (dataset_name, side, 86))
    # model.load_image_model(image_model_path)

    # audio_model_path = model_root + ('final_gen_%s_%s_%s/ckpt/096.pth' % (dataset_name, 'upsample', side))
    # audio_model_path = model_root + ('final_gen_%s_%s_%s/ckpt/%03d.pth' % (dataset_name, 'upsample', side, 101))
    # model.load_audio_model(audio_model_path)
    # model.load_audio_cls(audio_model_path)

    # text_model_path = model_root + ('final_%s_%s/ckpt/%03d.pth' % (dataset_name, side, 151))
    # text_model_path = model_root + ('final_%s_%s/ckpt/final.pth' % (dataset_name, side))
    # model.load_text_model(text_model_path)

    # model.evaluation(media_loader, media)
    model.output(media_loader, media, args.recons_root, args.target_root)


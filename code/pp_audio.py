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

import utils
import models

class RealSideDataset(Dataset):
    def __init__(self, args, split):
        super(RealSideDataset).__init__()
        self.trace_c = args.trace_c
        self.trace_w = args.trace_w

        self.lms_dir = ('%s%s/' % (args.data_path[args.dataset]['media'], split))
        self.npz_dir = ('%s%s/' % (args.data_path[args.dataset]['pp-%s-%s' % (args.cpu, args.cache)], split))

        self.lms_list = sorted(os.listdir(self.lms_dir))
        self.npz_list = sorted(os.listdir(self.npz_dir))

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
        
        self.max_db = args.max_db
        self.min_db = args.min_db
        
        self.norm = transforms.Normalize((0.5,), (0.5,))
        # print('Max db: %f' % self.max_db)
        # print('Min db: %f' % self.min_db)
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
        trace = torch.from_numpy(trace).view([self.trace_c, self.trace_w, self.trace_w])

        # lms_name = self.lms_list[index]
        lms_name = npz_name.split('.')[0] + '.npz'
        assert npz_name.split('.')[0] == lms_name.split('.')[0]
        lms = np.load(self.lms_dir + lms_name)
        audio = lms['arr_0']
        audio = audio.astype(np.float32)
        audio = torch.from_numpy(audio).view([1, 128, 44]) # sc09
        audio = utils.my_scale(v=audio,
                               v_max=self.max_db,
                               v_min=self.min_db
                            )
        audio = self.norm(audio)

        content_name, ID_name = npz_name.split('_')[:2]
        content = int(self.content_label[content_name])
        ID = int(self.ID_label[ID_name])
        content = torch.LongTensor([content])
        ID = torch.LongTensor([ID])

        return trace, audio, content.squeeze(), ID.squeeze(), lms_name.split('.')[0]

class NoisyRealSideDataset(Dataset):
    def __init__(self, args, split):
        super(NoisyRealSideDataset).__init__()
        self.op = args.noise_pp_op
        self.k = args.noise_pp_k
        self.trace_c = args.trace_c
        self.trace_w = args.trace_w

        self.lms_dir = args.lms_dir + ('train/' if split == 'train' else 'test/')
        self.npz_dir = args.npz_dir + ('train/' if split == 'train' else 'test/')

        self.lms_list = sorted(os.listdir(self.lms_dir))
        self.npz_list = sorted(os.listdir(self.npz_dir))

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
        
        self.max_db = args.max_db
        self.min_db = args.min_db
        
        self.norm = transforms.Normalize((0.5,), (0.5,))
        # print('Max db: %f' % self.max_db)
        # print('Min db: %f' % self.min_db)
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
            # print('trace: ', trace.shape)
            del_num = int(length * self.k)
            del_index = np.random.choice(np.arange(length), del_num, replace=False)
            del_trace = np.delete(trace, del_index, axis=0)
            # print('del_trace: ', del_trace.shape)
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

        # lms_name = self.lms_list[index]
        lms_name = npz_name.split('.')[0] + '.npz'
        assert npz_name.split('.')[0] == lms_name.split('.')[0]
        lms = np.load(self.lms_dir + lms_name)
        audio = lms['arr_0']
        audio = audio.astype(np.float32)
        audio = torch.from_numpy(audio).view([1, 128, 44]) # sc09
        audio = utils.my_scale(v=audio,
                               v_max=self.max_db,
                               v_min=self.min_db
                            )
        audio = self.norm(audio)

        content_name, ID_name = npz_name.split('_')[:2]
        content = int(self.content_label[content_name])
        ID = int(self.ID_label[ID_name])
        content = torch.LongTensor([content])
        ID = torch.LongTensor([ID])

        return trace, audio, content.squeeze(), ID.squeeze(), lms_name.split('.')[0]

class AudioEngine(object):
    def __init__(self, args):
        self.args = args
        self.epoch = 0
        self.real_label = 1
        self.fake_label = 0
        self.mse = nn.MSELoss().to(self.args.device)
        self.l1 = nn.L1Loss().to(self.args.device)
        self.bce = nn.BCELoss().to(self.args.device)
        self.ce = nn.CrossEntropyLoss().to(self.args.device)
        self.init_model_optimizer()
        self.norm_inv = utils.NormalizeInverse((0.5,), (0.5,))

    def init_model_optimizer(self):
        print('Initializing Model and Optimizer...')
        self.enc = models.__dict__['trace_encoder_%d' % self.args.trace_w](dim=self.args.nz, nc=self.args.trace_c)
        #self.enc = models.__dict__['encoder_%d' % self.args.image_size](dim=self.args.nz, nc=self.args.nc)
        self.enc = self.enc.to(self.args.device)

        self.dec = models.__dict__['audio_decoder_128'](dim=self.args.nz, nc=1, out_s=44) # SC09 44, Sub-URMP 22
        self.dec = self.dec.to(self.args.device)

        self.optim = torch.optim.Adam(
                        list(self.enc.parameters()) + \
                        list(self.dec.parameters()),
                        lr=self.args.lr,
                        betas=(self.args.beta1, 0.999)
                        )

        self.E = models.__dict__['%s_embed_%d' % (self.args.dataset.replace('-', '_'), self.args.lms_w)](dim=self.args.nz, nc=1)
        self.E = self.E.to(self.args.device)

        self.D = models.__dict__['classifier'](dim=self.args.nz, n_class=1)
        self.D = self.D.to(self.args.device)

        self.C1 = models.__dict__['classifier'](dim=self.args.nz, n_class=self.args.num_ID, use_bn=True)
        self.C1 = self.C1.to(self.args.device)

        self.C2 = models.__dict__['classifier'](dim=self.args.nz, n_class=self.args.num_content, use_bn=True)
        self.C2 = self.C2.to(self.args.device)

        self.optim_D = torch.optim.Adam(
                        list(self.E.parameters()) + \
                        list(self.D.parameters()) + \
                        list(self.C1.parameters())+ \
                        list(self.C2.parameters()),
                        lr=self.args.lr,
                        betas=(self.args.beta1, 0.999)
                        )

    def save_model(self, path):
        print('Saving Model on %s ...' % (path))
        state = {
            'enc': self.enc.state_dict(),
            'dec': self.dec.state_dict(),
            'E': self.E.state_dict(),
            'D': self.D.state_dict(),
            'C1': self.C1.state_dict(),
            'C2': self.C2.state_dict()
        }
        torch.save(state, path)

    def load_model(self, path):
        print('Loading Model from %s ...' % (path))
        ckpt = torch.load(path, map_location=self.args.device)
        self.enc.load_state_dict(ckpt['enc'])
        self.dec.load_state_dict(ckpt['dec'])
        self.E.load_state_dict(ckpt['E'])
        self.D.load_state_dict(ckpt['D'])
        self.C1.load_state_dict(ckpt['C1'])
        self.C2.load_state_dict(ckpt['C2'])

    def save_output(self, output, path):
        for i in range(len(output)):
            output[i] = self.norm_inv(output[i])
        output = utils.my_scale_inv(v=output, 
                                        v_max=self.args.max_db,
                                        v_min=self.args.min_db)
        output = output.cpu().numpy()
        np.savez_compressed(path, output)

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

    def zero_grad_G(self):
        self.enc.zero_grad()
        self.dec.zero_grad()

    def zero_grad_D(self):
        self.E.zero_grad()
        self.D.zero_grad()
        self.C1.zero_grad()
        self.C2.zero_grad()

    def set_train(self):
        self.enc.train()
        self.dec.train()
        self.E.train()
        self.D.train()
        self.C1.train()
        self.C2.train()

    def set_eval(self):
        self.enc.eval()
        self.dec.eval()
        self.E.eval()
        self.D.eval()
        self.C1.eval()
        self.C2.eval()  

    def train(self, data_loader):
        with torch.autograd.set_detect_anomaly(True):
            self.epoch += 1
            self.set_train()
            record = utils.Record()
            record_G = utils.Record()
            record_D = utils.Record()
            record_C1_real = utils.Record() # C1 for ID
            record_C1_fake = utils.Record()
            record_C2_real = utils.Record() # C2 for content
            record_C2_fake = utils.Record()
            record_C1_real_acc = utils.Record() # C1 for ID
            record_C1_fake_acc = utils.Record()
            record_C2_real_acc = utils.Record() # C2 for content
            record_C2_fake_acc = utils.Record()
            start_time = time.time()
            progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start()
            # progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
            for i, (trace, image, content, ID, name_list) in enumerate(data_loader):
                progress.update(i + 1)
                image = image.to(self.args.device)
                trace = trace.to(self.args.device)
                content = content.to(self.args.device)
                ID = ID.to(self.args.device)
                bs = image.size(0)

                # train D with real
                self.zero_grad_D()
                real_data = image.to(self.args.device)
                batch_size = real_data.size(0)
                label_real = torch.full((batch_size, 1), self.real_label, dtype=real_data.dtype).to(self.args.device)
                label_fake = torch.full((batch_size, 1), self.fake_label, dtype=real_data.dtype).to(self.args.device)

                embed_real = self.E(real_data)
                output_real = self.D(embed_real)
                errD_real = self.bce(output_real, label_real)
                D_x = output_real.mean().item()

                # train D with fake
                encoded = self.enc(trace)
                noise = torch.randn(bs, self.args.nz).to(self.args.device)
                decoded = self.dec(encoded + 0.05 * noise)
                
                output_fake = self.D(self.E(decoded.detach()))
                errD_fake = self.bce(output_fake, label_fake)
                D_G_z1 = output_fake.mean().item()
                
                errD = errD_real + errD_fake
                
                # train C with real
                pred1_real = self.C1(embed_real)
                errC1_real = self.ce(pred1_real, ID)

                pred2_real = self.C2(embed_real)
                errC2_real = self.ce(pred2_real, content)
                (errD_real + errD_fake + 10 * errC1_real + 10 * errC2_real).backward()

                self.optim_D.step()
                record_D.add(errD.item())
                record_C1_real.add(errC1_real.item())
                record_C2_real.add(errC2_real.item())

                record_C1_real_acc.add(utils.accuracy(pred1_real, ID))
                record_C2_real_acc.add(utils.accuracy(pred2_real, content))

                self.zero_grad_G()

                encoded = self.enc(trace)
                noise = torch.randn(bs, self.args.nz).to(self.args.device)
                decoded = self.dec(encoded + 0.05 * noise)

                embed_fake = self.E(decoded)
                label_real = torch.full((batch_size, 1), self.real_label, dtype=real_data.dtype).to(self.args.device)
                output_fake = self.D(embed_fake)
                pred1_fake = self.C1(embed_fake)
                pred2_fake = self.C2(embed_fake)

                errG = self.bce(output_fake, label_real)
                errC1_fake = self.ce(pred1_fake, ID)
                errC2_fake = self.ce(pred2_fake, content)
                recons_err = self.mse(decoded, image)

                (errG + 10 * errC1_fake + 10 * errC2_fake + self.args.lambd * recons_err).backward()
                D_G_z2 = output_fake.mean().item()
                self.optim.step()
                record_G.add(errG.item())
                record.add(recons_err.item())

                record_C1_fake.add(errC1_fake.item())
                record_C2_fake.add(errC2_fake.item())

                record_C1_fake_acc.add(utils.accuracy(pred1_fake, ID))
                record_C2_fake_acc.add(utils.accuracy(pred2_fake, content))
                    
            progress.finish()
            utils.clear_progressbar()
            print('----------------------------------------')
            print('Epoch: %d' % self.epoch)
            print('Costs Time: %.2f s' % (time.time() - start_time))
            print('Recons Loss: %f' % (record.mean()))
            print('Loss of G: %f' % (record_G.mean()))
            print('Loss of D: %f' % (record_D.mean()))
            print('Loss & Acc of C ID real: %f & %f' % (record_C1_real.mean(), record_C1_real_acc.mean()))
            print('Loss & Acc of C ID fake: %f & %f' % (record_C1_fake.mean(), record_C1_fake_acc.mean()))
            print('Loss & Acc of C content real: %f & %f' % (record_C2_real.mean(), record_C2_real_acc.mean()))
            print('Loss & Acc of C content fake: %f & %f' % (record_C2_fake.mean(), record_C2_fake_acc.mean()))
            print('D(x) is: %f, D(G(z1)) is: %f, D(G(z2)) is: %f' % (D_x, D_G_z1, D_G_z2))

    def test(self, data_loader):
        #with torch.autograd.set_detect_anomaly(True):
        self.set_eval()
        record = utils.Record()
        record_C1_fake_acc = utils.Record()
        record_C2_fake_acc = utils.Record()
        start_time = time.time()
        progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start()
        # progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
        with torch.no_grad():
            for i, (trace, image, content, ID, name_list) in enumerate(data_loader):
                progress.update(i + 1)
                image = image.to(self.args.device)
                trace = trace.to(self.args.device)
                content = content.to(self.args.device)
                ID = ID.to(self.args.device)
                bs = image.size(0)
                encoded = self.enc(trace)
                decoded = self.dec(encoded)

                embed_fake = self.E(decoded)
                pred1_fake = self.C1(embed_fake)
                pred2_fake = self.C2(embed_fake)

                record_C1_fake_acc.add(utils.accuracy(pred1_fake, ID))
                record_C2_fake_acc.add(utils.accuracy(pred2_fake, content))
                
                recons_err = self.mse(decoded, image)
                record.add(recons_err.item())
            progress.finish()
            utils.clear_progressbar()
            print('----------------------------------------')
            print('Test.')
            print('Costs Time: %.2f s' % (time.time() - start_time))
            print('Recons Loss: %f' % (record.mean()))
            print('Acc of C ID: %f' % record_C1_fake_acc.mean())
            print('Acc of C content: %f' % record_C2_fake_acc.mean())

    def inference(self, data_loader):
        self.set_eval()
        record = utils.Record()
        record_C1_fake_acc = utils.Record()
        record_C2_fake_acc = utils.Record()
        start_time = time.time()
        progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start()
        # progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
        with torch.no_grad():
            for i, (trace, image, content, ID, name_list) in enumerate(data_loader):
                progress.update(i + 1)
                image = image.to(self.args.device)
                trace = trace.to(self.args.device)
                content = content.to(self.args.device)
                ID = ID.to(self.args.device)
                bs = image.size(0)

                encoded = self.enc(trace)
                decoded = self.dec(encoded)
                
                embed_fake = self.E(decoded)
                pred1_fake = self.C1(embed_fake)
                pred2_fake = self.C2(embed_fake)

                record_C1_fake_acc.add(utils.accuracy(pred1_fake, ID))
                record_C2_fake_acc.add(utils.accuracy(pred2_fake, content))

                
                recons_err = self.mse(decoded, image)
                record.add(recons_err.item())

                self.save_audio(decoded, name_list, self.args.recons_root)
                self.save_audio(image, name_list, self.args.target_root)

            progress.finish()
            utils.clear_progressbar()
            print('----------------------------------------')
            print('Inference.')
            print('Costs Time: %.2f s' % (time.time() - start_time))
            print('Recons Loss: %f' % (record.mean()))
            print('ID Acc: %f' % (record_C1_fake_acc.mean()))
            print('Content Acc: %f' % (record_C2_fake_acc.mean()))



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
            (args.trace_w, args.trace_c) = (512, 8)
        elif args.cache == 'icache':
            (args.trace_w, args.trace_c) = (512, 2)
    elif args.cpu == 'intel':
        if args.cache == 'dcache':
            (args.trace_w, args.trace_c) = (512, 2)
        elif args.cache == 'icache':
            (args.trace_w, args.trace_c) = (512, 8)

    args.nz = 256
    print(args.exp_name)

    manual_seed = random.randint(1, 10000)
    print('Manual Seed: %d' % manual_seed)

    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    utils.make_path(args.output_root)
    utils.make_path(args.output_root + args.exp_name)

    args.lms_root = args.output_root + args.exp_name + '/lms/'
    args.ckpt_root = args.output_root + args.exp_name + '/ckpt/'
    args.audio_root = args.output_root + args.exp_name + '/audio/'
    args.recons_root = args.output_root + args.exp_name + '/recons/'
    args.target_root = args.output_root + args.exp_name + '/target/'

    utils.make_path(args.lms_root)
    utils.make_path(args.ckpt_root)
    utils.make_path(args.audio_root)
    utils.make_path(args.recons_root)
    utils.make_path(args.target_root)

    loader = DataLoader(args)

    train_dataset = RealSideDataset(args, split=args.data_path[args.dataset]['split'][0])
    args.num_content = train_dataset.content_cnt
    args.num_ID = train_dataset.ID_cnt
    
    test_dataset = RealSideDataset(args, split=args.data_path[args.dataset]['split'][1])

    engine = AudioEngine(args)

    train_loader = loader.get_loader(train_dataset)
    test_loader = loader.get_loader(test_dataset)

    # # Part B: for reconstructing media data
    # # B1. use our trained model
    # ROOT = '..' if os.environ.get('MANIFOLD_SCA') is None else os.environ.get('MANIFOLD_SCA')
    # engine.load_model(ROOT + '/models/pp/SC09_intel_dcache/final.pth')
    
    # # # B2. use your model
    # # # engine.load_model(args.ckpt_root + 'final.pth')
    
    # engine.inference(test_loader)

    #############################################
    # If you want to approximate manifold,      #
    # comment `Part B` and uncomment `Part A`.  #
    # If you want to reconstruct media data     #
    # from unknown side channel records,        #
    # comment `Part A` and uncomment `Part B`   #
    #############################################

    # Part A: for training
    for i in range(engine.epoch, args.num_epoch):
        engine.train(train_loader)
        if i % args.test_freq == 0:
            engine.test(test_loader)
            engine.save_model((args.ckpt_root + '%03d.pth') % (i + 1))
    engine.save_model((args.ckpt_root + 'final.pth'))
    
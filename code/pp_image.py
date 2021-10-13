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
        self.img_dir = ('%s%s/' % (args.data_path[args.dataset]['media'], split))
        self.npz_dir = ('%s%s/' % (args.data_path[args.dataset]['pp-%s-%s' % (args.cpu, args.cache)], split))

        self.npz_list = sorted(os.listdir(self.npz_dir))[:80000]
        self.img_list = sorted(os.listdir(self.img_dir))[:80000]

        self.transform = transforms.Compose([
                       transforms.Resize(args.image_size),
                       transforms.CenterCrop(args.image_size),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
               ])
        
        print('Total %d Data Points.' % len(self.npz_list))

        json_path = args.data_path[args.dataset]['ID_path']
        with open(json_path, 'r') as f:
            self.ID_dict = json.load(f)

        self.ID_cnt = len(set(self.ID_dict.values()))
        print('Total %d ID.' % self.ID_cnt)

    def __len__(self):
        return len(self.npz_list)

    def __getitem__(self, index):
        npz_name = self.npz_list[index]
        prefix = npz_name.split('.')[0]
        img_name = prefix + '.jpg'
        ID = int(self.ID_dict[img_name]) - 1

        npz = np.load(self.npz_dir + npz_name)
        trace = npz['arr_0']
        trace = trace.astype(np.float32)
        trace = torch.from_numpy(trace).view([self.trace_c, self.trace_w, self.trace_w])

        image = Image.open(self.img_dir + img_name)
        image = self.transform(image)

        ID = torch.LongTensor([ID]).squeeze()

        return trace, image, prefix, ID

class NoisyRealSideDataset(Dataset):
    def __init__(self, args, split):
        super(NoisyRealSideDataset).__init__()
        self.op = args.noise_pp_op
        self.k = args.noise_pp_k
        self.trace_c = args.trace_c
        self.trace_w = args.trace_w
        self.npz_dir = args.npz_dir + ('train/' if split == 'train' else 'test/')
        self.img_dir = args.image_dir + ('train/' if split == 'train' else 'test/')

        self.npz_list = sorted(os.listdir(self.npz_dir))
        self.img_list = sorted(os.listdir(self.img_dir))

        self.transform = transforms.Compose([
                       transforms.Resize(args.image_size),
                       transforms.CenterCrop(args.image_size),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
               ])
        
        print('Total %d Data Points.' % len(self.npz_list))

    def __len__(self):
        return len(self.npz_list)

    def __getitem__(self, index):
        npz_name = self.npz_list[index]
        # img_name = self.img_list[index]
        # npz_name = img_name.split('.')[0] + '.npz'
        prefix = npz_name.split('.')[0]
        img_name = prefix + '.jpg'

        npz = np.load(self.npz_dir + npz_name)
        trace = npz['arr_0']
        trace = trace.astype(np.float32)

        if self.op == 'order':
            assert self.k in [10, 100, 500]
            length = len(trace)
            exchange_index = np.random.choice(np.arange(length), self.k, replace=False)
            for ex_i in range(len(exchange_index)-1):
                ex_j = ex_i + 1
                trace[[ex_i, ex_j]] = trace[[ex_j, ex_i]]
        if self.op == 'out':
            assert self.k in [0.2, 0.5]
            (length, n_set) = trace.shape
            del_num = int(length * self.k)
            del_index = np.random.choice(np.arange(length), del_num, replace=False)
            del_trace = np.delete(trace, del_index, axis=0)
            trace = np.concatenate([del_trace, np.zeros((del_num, n_set))])
            trace = trace.astype(np.float32)

        trace = torch.from_numpy(trace).view([self.trace_c, self.trace_w, self.trace_w])

        if self.op == 'flip':
            assert self.k in [0.2, 0.5]
            fliped = 1 - trace
            flip_mask = torch.ones(trace.size())
            flip_mask = F.dropout(flip_mask, p=(1-self.k))
            keep_mask = 1 - flip_mask
            trace = flip_mask * fliped + keep_mask * trace

        image = Image.open(self.img_dir + img_name)
        image = self.transform(image)

        return trace, image, prefix


class ImageEngine(object):
    def __init__(self, args):
        self.args = args
        self.epoch = 0
        self.mse = nn.MSELoss().to(self.args.device)
        self.l1 = nn.L1Loss().to(self.args.device)
        self.bce = nn.BCELoss().to(self.args.device)
        self.ce = nn.CrossEntropyLoss().to(self.args.device)
        self.real_label = 1
        self.fake_label = 0
        self.init_model_optimizer()
        if self.args.use_refiner:
            self.init_refiner_optimizer()
        else:
            self.init_indicator_optimizer()

    def init_model_optimizer(self):
        print('Initializing Model and Optimizer...')
        self.enc = models.__dict__['trace_encoder_%d' % self.args.trace_w](dim=self.args.nz, nc=args.trace_c)
        self.enc = self.enc.to(self.args.device)

        self.dec = models.__dict__['ResDecoder128'](dim=self.args.nz, nc=3)
        # self.dec = models.__dict__['image_decoder_%d' % self.args.image_size](dim=self.args.nz, nc=self.args.nc)
        self.dec = self.dec.to(self.args.device)    

        self.optim = torch.optim.Adam(
                        list(self.enc.parameters()) + \
                        list(self.dec.parameters()),
                        lr=self.args.lr,
                        betas=(self.args.beta1, 0.999)
                        )

    def init_indicator_optimizer(self):
        self.E = models.__dict__['image_output_embed_128'](dim=self.args.nz, nc=3)
        self.E = self.E.to(self.args.device)

        self.D = models.__dict__['classifier'](dim=self.args.nz, n_class=1, use_bn=False)
        self.D = self.D.to(self.args.device)

        self.C = models.__dict__['classifier'](dim=self.args.nz, n_class=self.args.n_class, use_bn=False)
        self.C = self.C.to(self.args.device)

        self.optim_D = torch.optim.Adam(
                        list(self.E.parameters()) + \
                        list(self.D.parameters()) + \
                        list(self.C.parameters()),
                        lr=self.args.lr,
                        betas=(self.args.beta1, 0.999)
                        )

    def init_refiner_optimizer(self):
        self.G = models.__dict__['RefinerG_BN'](nc=3, ngf=self.args.nz)
        self.G = self.G.to(self.args.device)

        self.D = models.__dict__['RefinerD'](nc=3, ndf=self.args.nz)
        self.D = self.D.to(self.args.device)

        self.optim_G = torch.optim.Adam(
                        self.G.parameters(),
                        lr=self.args.lr,
                        betas=(self.args.beta1, 0.999)
                        )
        
        self.optim_D = torch.optim.Adam(
                        self.D.parameters(),
                        lr=self.args.lr,
                        betas=(self.args.beta1, 0.999)
                        )

    def save_model(self, path):
        print('Saving Model in %s ...' % (path))
        state = {
            'enc': self.enc.state_dict(),
            'dec': self.dec.state_dict(),
        }
        if not self.args.use_refiner:
            state.update({
                'E': self.E.state_dict(),
                'D': self.D.state_dict(),
                'C': self.C.state_dict()
            })
        torch.save(state, path)

    def save_refiner(self, path):
        print('Saving Refiner in %s ...' % (path))
        state = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict()
        }
        torch.save(state, path)

    def load_model(self, path):
        print('Loading Model from %s ...' % (path))
        ckpt = torch.load(path, map_location=self.args.device)
        self.enc.load_state_dict(ckpt['enc'])
        self.dec.load_state_dict(ckpt['dec'])
        if not self.args.use_refiner:
            self.E.load_state_dict(ckpt['E'])
            self.D.load_state_dict(ckpt['D'])
            self.C.load_state_dict(ckpt['C'])

    def load_refiner(self, path):
        print('Loading Refiner from %s ...' % (path))
        ckpt = torch.load(path, map_location=self.args.device)
        self.G.load_state_dict(ckpt['G'])
        self.D.load_state_dict(ckpt['D'])

    def load_encoder(self, path):
        print('Loading Encoder from %s ...' % (path))
        ckpt = torch.load(path, map_location=self.args.device)
        self.enc.load_state_dict(ckpt['enc'])

    def save_output(self, output, path):
        utils.save_image(output.data, path, normalize=True)

    def save_image(self, output, name_list, path):
        assert len(output) == len(name_list)
        for i in range(len(output)):
            utils.save_image(output[i].unsqueeze(0).data,
                             path + name_list[i] + '.jpg',
                             normalize=True, nrow=1, padding=0)

    def zero_grad_G(self):
        self.enc.zero_grad()
        self.dec.zero_grad()
        
    def zero_grad_D(self):
        self.E.zero_grad()
        self.D.zero_grad()
        self.C.zero_grad()

    def set_train(self):
        self.enc.train()
        self.dec.train()
        self.E.train()
        self.D.train()
        self.C.train()

    def set_eval(self):
        self.enc.eval()
        self.dec.eval()
        self.E.eval()
        self.D.eval()
        self.C.eval()

    def train_indicator(self, data_loader):
        with torch.autograd.set_detect_anomaly(True):
            self.epoch += 1
            self.set_train()
            record = utils.Record()
            record_G = utils.Record()
            record_D = utils.Record()
            record_C_real = utils.Record() # C1 for ID
            record_C_fake = utils.Record()
            record_C_real_acc = utils.Record() # C1 for ID
            record_C_fake_acc = utils.Record()
            start_time = time.time()
            progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start()
            # progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
            for i, (trace, image, prefix, ID) in enumerate(data_loader):
                progress.update(i + 1)
                image = image.to(self.args.device)
                trace = trace.to(self.args.device)
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
                pred_real = self.C(embed_real)
                errC_real = self.ce(pred_real, ID)
                (errD_real + errD_fake + errC_real).backward()

                self.optim_D.step()
                record_D.add(errD)
                record_C_real.add(errC_real)

                record_C_real_acc.add(utils.accuracy(pred_real, ID))

                # train G with D and C
                self.zero_grad_G()

                encoded = self.enc(trace)
                noise = torch.randn(bs, self.args.nz).to(self.args.device)
                decoded = self.dec(encoded + 0.05 * noise)

                embed_fake = self.E(decoded)
                label_real = torch.full((batch_size, 1), self.real_label, dtype=real_data.dtype).to(self.args.device)
                output_fake = self.D(embed_fake)
                pred_fake = self.C(embed_fake)

                errG = self.bce(output_fake, label_real)
                errC_fake = self.ce(pred_fake, ID)
                recons_err = self.mse(decoded, image)

                (errG + errC_fake + self.args.lambd * recons_err).backward()
                D_G_z2 = output_fake.mean().item()
                self.optim.step()
                record_G.add(errG)
                record.add(recons_err.item())
                record_C_fake.add(errC_fake)
                record_C_fake_acc.add(utils.accuracy(pred_fake, ID))

            progress.finish()
            utils.clear_progressbar()
            print('----------------------------------------')
            print('Epoch: %d' % self.epoch)
            print('Costs Time: %.2f s' % (time.time() - start_time))
            print('Recons Loss: %f' % (record.mean()))
            print('Loss of G: %f' % (record_G.mean()))
            print('Loss of D: %f' % (record_D.mean()))
            print('Loss & Acc of C ID real: %f & %f' % (record_C_real.mean(), record_C_real_acc.mean()))
            print('Loss & Acc of C ID fake: %f & %f' % (record_C_fake.mean(), record_C_fake_acc.mean()))
            print('D(x) is: %f, D(G(z1)) is: %f, D(G(z2)) is: %f' % (D_x, D_G_z1, D_G_z2))
            self.save_output(decoded, (self.args.image_root + 'train_%03d.jpg') % self.epoch)
            self.save_output(image, (self.args.image_root + 'train_%03d_target.jpg') % self.epoch)

    def pre_train(self, data_loader):
        with torch.autograd.set_detect_anomaly(True):
            self.epoch += 1
            self.enc.train()
            self.dec.train()
            record = utils.Record()
            start_time = time.time()
            progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start()
            # progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
            for i, (trace, image, prefix, ID) in enumerate(data_loader):
                progress.update(i + 1)
                image = image.to(self.args.device)
                trace = trace.to(self.args.device)
                bs = image.size(0)

                self.enc.zero_grad()
                self.dec.zero_grad()

                encoded = self.enc(trace)
                decoded = self.dec(encoded)
                
                recons_err = self.l1(decoded, image)
                recons_err.backward()
                self.optim.step()
                record.add(recons_err.item())
            progress.finish()
            utils.clear_progressbar()
            print('----------------------------------------')
            print('Epoch: %d' % self.epoch)
            print('Costs Time: %.2f s' % (time.time() - start_time))
            print('Recons Loss: %f' % (record.mean()))
            self.save_output(decoded, (self.args.image_root + 'train_%03d.jpg') % self.epoch)
            self.save_output(image, (self.args.image_root + 'train_%03d_target.jpg') % self.epoch)

    def train_refiner(self, data_loader):
        with torch.autograd.set_detect_anomaly(True):
            self.epoch += 1
            self.G.train()
            self.D.train()
            record = utils.Record()
            record_G = utils.Record()
            record_D = utils.Record()
            start_time = time.time()
            progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start()
            # progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
            for i, (trace, image, prefix, ID) in enumerate(data_loader):
                progress.update(i + 1)
                image = image.to(self.args.device)
                trace = trace.to(self.args.device)
                bs = image.size(0)

                # train D with real
                self.D.zero_grad()
                real_data = image.to(self.args.device)
                batch_size = real_data.size(0)
                label_real = torch.full((batch_size, 1), self.real_label, dtype=real_data.dtype).to(self.args.device)
                label_fake = torch.full((batch_size, 1), self.fake_label, dtype=real_data.dtype).to(self.args.device)

                output_real = self.D(real_data)
                errD_real = self.bce(output_real, label_real)
                D_x = output_real.mean().item()

                # train D with fake
                with torch.no_grad(): decoded = self.dec(self.enc(trace))
                fake_data = self.G(decoded)
                output_fake = self.D(fake_data.detach())
                errD_fake = self.bce(output_fake, label_fake)
                D_G_z1 = output_fake.mean().item()
                
                errD = errD_real + errD_fake
                errD.backward()

                self.optim_D.step()
                record_D.add(errD)

                # train G with D and C
                self.G.zero_grad()

                with torch.no_grad(): decoded = self.dec(self.enc(trace))
                fake_data = self.G(decoded)
                
                label_real = torch.full((batch_size, 1), self.real_label, dtype=real_data.dtype).to(self.args.device)
                output_fake = self.D(fake_data)

                errG = self.bce(output_fake, label_real)
                recons_err = self.l1(fake_data, decoded)

                (errG + self.args.lambd * recons_err).backward()
                D_G_z2 = output_fake.mean().item()
                self.optim_G.step()
                record_G.add(errG)
                record.add(recons_err.item())

            progress.finish()
            utils.clear_progressbar()
            print('----------------------------------------')
            print('Epoch: %d' % self.epoch)
            print('Costs Time: %.2f s' % (time.time() - start_time))
            print('Recons Loss: %f' % (record.mean()))
            print('Loss of G: %f' % (record_G.mean()))
            print('Loss of D: %f' % (record_D.mean()))
            print('D(x) is: %f, D(G(z1)) is: %f, D(G(z2)) is: %f' % (D_x, D_G_z1, D_G_z2))
            self.save_output(decoded, (self.args.image_root + 'train_%03d_decoded.jpg') % self.epoch)
            self.save_output(fake_data, (self.args.image_root + 'train_%03d_fake.jpg') % self.epoch)
            self.save_output(image, (self.args.image_root + 'train_%03d_target.jpg') % self.epoch)
            
    def fit(self, train_loader, test_loader):
        if self.args.use_refiner:
            for i in range(self.epoch, self.args.num_epoch):
                self.pre_train(train_loader)
                if i % self.args.test_freq == 0:
                    self.test(test_loader)
                    self.save_model((self.args.ckpt_root + '%03d.pth') % (i + 1))
            self.save_model((self.args.ckpt_root + 'final.pth'))
            self.epoch = 0
            for i in range(self.epoch, self.args.num_epoch):
                self.train_refiner(train_loader)
                if i % self.args.test_freq == 0:
                    self.save_refiner((self.args.ckpt_root + 'refiner-%03d.pth') % (i + 1))
            self.save_refiner((self.args.ckpt_root + 'refiner-final.pth'))
        else:
            for i in range(self.epoch, self.args.num_epoch):
                self.train_indicator(train_loader)
                if i % self.args.test_freq == 0:
                    self.test(test_loader)
                    self.save_model((self.args.ckpt_root + '%03d.pth') % (i + 1))
            self.save_model((self.args.ckpt_root + 'final.pth'))

    def test(self, data_loader):
        #with torch.autograd.set_detect_anomaly(True):
        record = utils.Record()
        start_time = time.time()
        progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start()
        # progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
        with torch.no_grad():
            for i, (trace, image, prefix, ID) in enumerate(data_loader):
                progress.update(i + 1)
                image = image.to(self.args.device)
                trace = trace.to(self.args.device)
                ID = ID.to(self.args.device)
                bs = image.size(0)
                encoded = self.enc(trace)
                decoded = self.dec(encoded)

                recons_err = self.l1(decoded, image)
                record.add(recons_err.item())
            progress.finish()
            utils.clear_progressbar()
            print('----------------------------------------')
            print('Test.')
            print('Costs Time: %.2f s' % (time.time() - start_time))
            print('Recons Loss: %f' % (record.mean()))
            self.save_output(decoded, (self.args.image_root + 'test_%03d.jpg') % self.epoch)
            self.save_output(image, (self.args.image_root + 'test_%03d_target.jpg') % self.epoch)

    def inference(self, data_loader, sub):
        record = utils.Record()
        start_time = time.time()
        recons_dir = '%s%s/' % (self.args.recons_root, sub)
        utils.make_path(recons_dir)
        target_dir = '%s%s/' % (self.args.target_root, sub)
        utils.make_path(target_dir)
        progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start()
        # progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
        with torch.no_grad():
            for i, (trace, image, name_list) in enumerate(data_loader):
                progress.update(i + 1)
                image = image.to(self.args.device)
                trace = trace.to(self.args.device)

                bs = image.size(0)
                encoded = self.enc(trace)
                decoded = self.dec(encoded)
                if self.args.use_refiner:
                    decoded = self.G(decoded)
                recons_err = self.l1(decoded, image)
                record.add(recons_err.item())
                self.save_image(decoded, name_list, recons_dir)
                self.save_image(image, name_list, target_dir)
            progress.finish()
            utils.clear_progressbar()
            print('----------------------------------------')
            print('Inference.')
            print('Costs Time: %.2f s' % (time.time() - start_time))
            print('Recons Loss: %f' % (record.mean()))

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
            (args.trace_w, args.trace_c) = (256, 4)
        elif args.cache == 'icache':
            (args.trace_w, args.trace_c) = (256, 1)
    elif args.cpu == 'intel':
        if args.cache == 'dcache':
            (args.trace_w, args.trace_c) = (256, 1)
        elif args.cache == 'icache':
            (args.trace_w, args.trace_c) = (256, 1)

    args.nz = 128
    print(args.exp_name)
    args.use_refiner = 1

    manual_seed = random.randint(1, 10000)
    print('Manual Seed: %d' % manual_seed)

    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    utils.make_path(args.output_root)
    utils.make_path(args.output_root + args.exp_name)

    args.image_root = args.output_root + args.exp_name + '/image/'
    args.ckpt_root = args.output_root + args.exp_name + '/ckpt/'
    args.recons_root = args.output_root + args.exp_name + '/recons/'
    args.target_root = args.output_root + args.exp_name + '/target/'

    utils.make_path(args.image_root)
    utils.make_path(args.ckpt_root)
    utils.make_path(args.recons_root)
    utils.make_path(args.target_root)

    loader = DataLoader(args)

    train_dataset = RealSideDataset(args, split=args.data_path[args.dataset]['split'][0])
    test_dataset = RealSideDataset(args, split=args.data_path[args.dataset]['split'][1])
    args.n_class = train_dataset.ID_cnt

    engine = ImageEngine(args)

    train_loader = loader.get_loader(train_dataset)
    test_loader = loader.get_loader(test_dataset)

    # # Part B: for reconstructing media data
    
    # # B1. use our trained model
    # ROOT = '..' if os.environ.get('MANIFOLD_SCA') is None else os.environ.get('MANIFOLD_SCA') # use our trained model
    # engine.load_model(ROOT + '/models/pp/CelebA_intel_dcache/final.pth')
    # if arg.use_refiner:
    #     engine.load_refiner(ROOT + '/models/pin/CelebA_refiner/refiner-final.pth')
    
    # # # B2. use your model
    # # engine.load_model(args.ckpt_root + 'final.pth')
    # # if arg.use_refiner:
    # #     engine.load_refiner(args.ckpt_root + 'refiner-final.pth')
    
    # engine.inference(test_loader, 'test')
    
    #############################################
    # If you want to approximate manifold,      #
    # comment `Part B` and uncomment `Part A`.  #
    # If you want to reconstruct media data     #
    # from unknown side channel records,        #
    # comment `Part A` and uncomment `Part B`   #
    #############################################

    # Part A: for approximating manifold
    engine.fit(train_loader, test_loader)
       
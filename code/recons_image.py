import time
import numpy as np
import progressbar

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import models

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

    def init_model_optimizer(self):
        print('Initializing Model and Optimizer...')
        self.enc = models.__dict__['attn_trace_encoder_%d' % self.args.trace_w](dim=self.args.nz, nc=self.args.trace_c)
        self.enc = self.enc.to(self.args.device)

        self.dec = models.__dict__['ResDecoder%d' % self.args.image_size](dim=self.args.nz, nc=self.args.nc)
        # self.dec = models.__dict__['image_decoder_%d' % self.args.image_size](dim=self.args.nz, nc=self.args.nc)
        self.dec = self.dec.to(self.args.device)    

        self.optim = torch.optim.Adam(
                        list(self.enc.parameters()) + \
                        list(self.dec.parameters()),
                        lr=self.args.lr,
                        betas=(self.args.beta1, 0.999)
                        )

        self.E = models.__dict__['image_output_embed_128'](dim=self.args.nz, nc=self.args.nc)
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

    def save_model(self, path):
        print('Saving Model on %s ...' % (path))
        state = {
            'enc': self.enc.state_dict(),
            'dec': self.dec.state_dict(),
            'E': self.E.state_dict(),
            'D': self.D.state_dict(),
            'C': self.C.state_dict()
        }
        torch.save(state, path)

    def load_model(self, path):
        print('Loading Model from %s ...' % (path))
        ckpt = torch.load(path, map_location=self.args.device)
        self.enc.load_state_dict(ckpt['enc'])
        self.dec.load_state_dict(ckpt['dec'])
        self.E.load_state_dict(ckpt['E'])
        self.D.load_state_dict(ckpt['D'])
        self.C.load_state_dict(ckpt['C'])

    def save_output(self, output, path):
        utils.save_image(output.data, path, normalize=True)

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

    def train(self, data_loader):
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
            #progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start()
            progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
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
            
    def test(self, data_loader):
        #with torch.autograd.set_detect_anomaly(True):
        self.set_eval()
        record = utils.Record()
        start_time = time.time()
        #progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start()
        progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
        with torch.no_grad():
            for i, (trace, image, prefix, ID) in enumerate(data_loader):
                progress.update(i + 1)
                image = image.to(self.args.device)
                trace = trace.to(self.args.device)
                ID = ID.to(self.args.device)
                bs = image.size(0)
                encoded = self.enc(trace)
                decoded = self.dec(encoded)                
                recons_err = self.mse(decoded, image)
                record.add(recons_err.item())
            progress.finish()
            utils.clear_progressbar()
            print('----------------------------------------')
            print('Test.')
            print('Costs Time: %.2f s' % (time.time() - start_time))
            print('Recons Loss: %f' % (record.mean()))
            self.save_output(decoded, (self.args.image_root + 'test_%03d.jpg') % self.epoch)
            self.save_output(image, (self.args.image_root + 'test_%03d_target.jpg') % self.epoch)

if __name__ == '__main__':
    import argparse
    import random

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
    args.trace_w = 256
    args.nz = 128

    print(args.exp_name)
    manual_seed = random.randint(1, 10000)
    print('Manual Seed: %d' % manual_seed)

    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    utils.make_path(args.output_root)
    utils.make_path(args.output_root + args.exp_name)

    args.image_root = args.output_root + args.exp_name + '/image/'
    args.ckpt_root = args.output_root + args.exp_name + '/ckpt/'

    utils.make_path(args.image_root)
    utils.make_path(args.ckpt_root)

    loader = DataLoader(args)

    assert args.dataset == 'CelebA'
    train_dataset = CelebaDataset(
                    img_dir=args.data_path[args.dataset]['media'], 
                    npz_dir=args.data_path[args.dataset][args.side],
                    ID_path=args.data_path[args.dataset]['ID_path'],
                    split=args.data_path[args.dataset]['split'][0],
                    trace_c=args.trace_c,
                    trace_w=args.trace_w,
                    image_size=args.image_size,
                    side=args.side
                )
    args.n_class = train_dataset.ID_cnt

    test_dataset = CelebaDataset(
                    img_dir=args.data_path[args.dataset]['media'], 
                    npz_dir=args.data_path[args.dataset][args.side],
                    ID_path=args.data_path[args.dataset]['ID_path'],
                    split=args.data_path[args.dataset]['split'][1],
                    trace_c=args.trace_c,
                    trace_w=args.trace_w,
                    image_size=args.image_size,
                    side=args.side
                )

    engine = ImageEngine(args)

    train_loader = loader.get_loader(train_dataset)
    test_loader = loader.get_loader(test_dataset)

    for i in range(args.num_epoch):
        engine.train(train_loader)
        if i % args.test_freq == 0:
            engine.test(test_loader)
            engine.save_model((args.ckpt_root + '%03d.pth') % (i + 1))
    engine.save_model((args.ckpt_root + 'final.pth'))
import time
import numpy as np
import progressbar

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import models

class Im(object):
    def __init__(self, args):
        self.args = args
        self.epoch = 0
        self.mse = nn.MSELoss().cuda()
        self.l1 = nn.L1Loss().cuda()
        self.bce = nn.BCELoss().cuda()
        self.ce = nn.CrossEntropyLoss().cuda()
        self.real_label = 1
        self.fake_label = 0
        self.init_model_optimizer()

    def init_model_optimizer(self):
        print('Initializing Model and Optimizer...')
        self.enc = models.__dict__['attn_trace_encoder_256'](dim=self.args.nz, nc=6)
        self.enc = torch.nn.DataParallel(self.enc).cuda()

        self.dec = models.__dict__['ResDecoder128'](dim=self.args.nz, nc=3)
        self.dec = torch.nn.DataParallel(self.dec).cuda()    

        self.optim = torch.optim.Adam(
                        list(self.enc.module.parameters()) + \
                        list(self.dec.module.parameters()),
                        lr=self.args.lr,
                        betas=(self.args.beta1, 0.999)
                        )

        self.E = models.__dict__['image_output_embed_128'](dim=self.args.nz, nc=3)
        self.E = torch.nn.DataParallel(self.E).cuda()

        self.D = models.__dict__['classifier'](dim=self.args.nz, n_class=1, use_bn=False)
        self.D = torch.nn.DataParallel(self.D).cuda()

        self.C = models.__dict__['classifier'](dim=self.args.nz, n_class=self.args.n_class, use_bn=False)
        self.C = torch.nn.DataParallel(self.C).cuda()

        self.optim_D = torch.optim.Adam(
                        list(self.E.module.parameters()) + \
                        list(self.D.module.parameters()) + \
                        list(self.C.module.parameters()),
                        lr=self.args.D_lr,
                        betas=(self.args.beta1, 0.999)
                        )

    def save_model(self, path):
        print('Saving Model on %s ...' % (path))
        state = {
            'enc': self.enc.module.state_dict(),
            'dec': self.dec.module.state_dict(),
            'E': self.E.module.state_dict(),
            'D': self.D.module.state_dict(),
            'C': self.C.module.state_dict()
        }
        torch.save(state, path)

    def load_model(self, path):
        print('Loading Model from %s ...' % (path))
        ckpt = torch.load(path)
        self.enc.module.load_state_dict(ckpt['enc'])
        self.dec.module.load_state_dict(ckpt['dec'])
        self.E.module.load_state_dict(ckpt['E'])
        self.D.module.load_state_dict(ckpt['D'])
        self.C.module.load_state_dict(ckpt['C'])

    def load_encoder(self, path):
        print('Loading Encoder from %s ...' % (path))
        ckpt = torch.load(path)
        self.enc.module.load_state_dict(ckpt['enc'])

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
        #self.enc.eval()
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
            for i, (trace, image, ID) in enumerate(data_loader):
                progress.update(i + 1)
                image = image.cuda()
                trace = trace.cuda()
                ID = ID.cuda()
                bs = image.size(0)
                #trace = torch.randn([bs, 2, 256, 256]).cuda()

                # train D with real
                self.zero_grad_D()
                real_data = image.cuda()
                batch_size = real_data.size(0)
                label_real = torch.full((batch_size, 1), self.real_label, dtype=real_data.dtype).cuda()
                label_fake = torch.full((batch_size, 1), self.fake_label, dtype=real_data.dtype).cuda()

                embed_real = self.E(real_data)
                output_real = self.D(embed_real)
                errD_real = self.bce(output_real, label_real)
                D_x = output_real.mean().item()

                # train D with fake
                encoded = self.enc(trace)
                #noise = torch.randn(bs, self.args.ng).cuda()
                #decoded = self.dec(torch.cat([encoded, noise], dim=1))
                noise = torch.randn(bs, self.args.nz).cuda()
                decoded = self.dec(encoded + 0.05 * noise)
                #embed_fake = self.E(decoded.detach())
                #label.fill_(self.fake_label)
                
                output_fake = self.D(self.E(decoded.detach()))
                errD_fake = self.bce(output_fake, label_fake)
                D_G_z1 = output_fake.mean().item()
                
                errD = errD_real + errD_fake
                
                # train C with real
                pred_real = self.C(embed_real)
                errC_real = self.ce(pred_real, ID)

                #errC_real.backward()
                (errD_real + errD_fake + errC_real).backward()

                self.optim_D.step()
                record_D.add(errD)
                record_C_real.add(errC_real)

                record_C_real_acc.add(utils.accuracy(pred_real, ID))

                for _ in range(1):
                    # train G with D and C
                    self.zero_grad_G()

                    encoded = self.enc(trace)
                    # noise = torch.randn(bs, self.args.ng).cuda()
                    # decoded = self.dec(torch.cat([encoded, noise], dim=1))
                    noise = torch.randn(bs, self.args.nz).cuda()
                    decoded = self.dec(encoded + 0.05 * noise)

                    embed_fake = self.E(decoded)
                    label_real = torch.full((batch_size, 1), self.real_label, dtype=real_data.dtype).cuda()
                    output_fake = self.D(embed_fake)
                    pred_fake = self.C(embed_fake)

                    errG = self.bce(output_fake, label_real)
                    errC_fake = self.ce(pred_fake, ID)
                    recons_err = self.mse(decoded, image)

                    (errG + errC_fake + 100 * recons_err).backward()
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
        record_C_fake_acc = utils.Record()
        start_time = time.time()
        #progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start()
        progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
        with torch.no_grad():
            for i, (trace, image, ID) in enumerate(data_loader):
                progress.update(i + 1)
                image = image.cuda()
                trace = trace.cuda()
                ID = ID.cuda()
                bs = image.size(0)
                #trace = torch.randn([bs, 2, 256, 256]).cuda()
                encoded = self.enc(trace)
                #noise = torch.randn(bs, self.args.ng).cuda()
                #decoded = self.dec(torch.cat([encoded, noise], dim=1))
                decoded = self.dec(encoded)

                embed_fake = self.E(decoded)
                pred_fake = self.C(embed_fake)

                record_C_fake_acc.add(utils.accuracy(pred_fake, ID))
                
                recons_err = self.mse(decoded, image)
                record.add(recons_err.item())
            progress.finish()
            utils.clear_progressbar()
            print('----------------------------------------')
            print('Test.')
            print('Costs Time: %.2f s' % (time.time() - start_time))
            print('Recons Loss: %f' % (record.mean()))
            print('Acc of C ID: %f' % record_C_fake_acc.mean())
            self.save_output(decoded, (self.args.image_root + 'test_%03d.jpg') % self.epoch)
            self.save_output(image, (self.args.image_root + 'test_%03d_target.jpg') % self.epoch)


if __name__ == '__main__':
    import argparse
    import random

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import utils
    from data_loader import *

    side = 'cacheline'

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default=('final_gen_celeba_%s' % side))

    parser.add_argument('--image_dir', type=str, default='/root/data/celeba_crop128/')
    parser.add_argument('--npz_dir', type=str, default=('/root/data/celeba_crop128_%s/' % side))
    parser.add_argument('--output_root', type=str, default='/root/output/SCA/')
    parser.add_argument('--image_size', type=int, default=128)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--D_lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--nz', type=int, default=128)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--test_freq', type=int, default=5)
    parser.add_argument('--n_class', type=int, default=-1)

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

    args.image_root = args.output_root + args.exp_name + '/image/'
    args.ckpt_root = args.output_root + args.exp_name + '/ckpt/'

    utils.make_path(args.image_root)
    utils.make_path(args.ckpt_root)

    with open(args.output_root + args.exp_name + '/args.json', 'w') as f:
        json.dump(args.__dict__, f)

    loader = DataLoader(args)

    train_dataset = CelebaDataset(
                    img_dir=args.image_dir, 
                    npz_dir=args.npz_dir, 
                    split='train',
                    image_size=args.image_size,
                    side=side
                )
    args.n_class = train_dataset.ID_cnt

    test_dataset = CelebaDataset(
                    img_dir=args.image_dir, 
                    npz_dir=args.npz_dir, 
                    split='test',
                    image_size=args.image_size,
                    side=side
                )

    model = Im(args)
    
    # model.load_encoder(encoder_path)
    #model.load_model(args.ckpt_root + '%03d.pth' % 76)

    train_loader = loader.get_loader(train_dataset)
    test_loader = loader.get_loader(test_dataset)

    for i in range(args.num_epoch):
        model.train(train_loader)
        if (i + 0) % args.test_freq == 0:
            model.test(test_loader)
            model.save_model((args.ckpt_root + '%03d.pth') % (i + 1))
    model.save_model((args.ckpt_root + 'final.pth'))
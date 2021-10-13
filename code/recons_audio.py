import time
import numpy as np
import progressbar

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import models

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
        self.enc = models.__dict__['attn_trace_encoder_%d' % self.args.trace_w](dim=self.args.nz, nc=self.args.trace_c)
        self.enc = self.enc.to(self.args.device)

        self.dec = models.__dict__['audio_decoder_%d' % self.args.lms_w](dim=self.args.nz, nc=1, out_s=self.args.lms_h)
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
            #progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start()
            progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
            for i, (trace, image, prefix, content, ID) in enumerate(data_loader):
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
                record_D.add(errD)
                record_C1_real.add(errC1_real)
                record_C2_real.add(errC2_real)

                record_C1_real_acc.add(utils.accuracy(pred1_real, ID))
                record_C2_real_acc.add(utils.accuracy(pred2_real, content))

                # train G with D and C
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
                record_G.add(errG)
                record.add(recons_err.item())

                record_C1_fake.add(errC1_fake)
                record_C2_fake.add(errC2_fake)

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
            self.save_output(decoded.detach(), (self.args.lms_root + 'train_%03d.npz') % self.epoch)
            self.save_output(image, (self.args.lms_root + 'train_%03d_target.npz') % self.epoch)

            #################################################
            # To assess the speaker IDs of SC09 audios,     #
            # we treat it as `face recognization` and       #
            # use the triplet loss in FaceNet. You can take #
            # `https://github.com/timesler/facenet-pytorch` #
            # for reference.                                #
            #################################################

    def test(self, data_loader):
        #with torch.autograd.set_detect_anomaly(True):
        self.set_eval()
        record = utils.Record()
        record_C1_fake_acc = utils.Record()
        record_C2_fake_acc = utils.Record()
        start_time = time.time()
        #progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start()
        progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
        with torch.no_grad():
            for i, (trace, image, prefix, content, ID) in enumerate(data_loader):
                progress.update(i + 1)
                image = image.to(self.args.device)
                trace = trace.to(self.args.device)
                content = content.to(self.args.device)
                ID = ID.to(self.args.device)
                bs = image.size(0)
                encoded = self.enc(trace)
                decoded = self.dec(encoded)

                embed_fake = self.E(decoded)
                # pred1_fake = self.C1(embed_fake)
                pred2_fake = self.C2(embed_fake)

                # record_C1_fake_acc.add(utils.accuracy(pred1_fake, ID))
                record_C2_fake_acc.add(utils.accuracy(pred2_fake, content))
                
                recons_err = self.mse(decoded, image)
                record.add(recons_err.item())
            progress.finish()
            utils.clear_progressbar()
            print('----------------------------------------')
            print('Test.')
            print('Costs Time: %.2f s' % (time.time() - start_time))
            print('Recons Loss: %f' % (record.mean()))
            # print('Acc of C ID: %f' % record_C1_fake_acc.mean())
            print('Acc of C content: %f' % record_C2_fake_acc.mean())
            self.save_output(decoded.detach(), (self.args.lms_root + 'test_%03d.npz') % self.epoch)
            self.save_output(image, (self.args.lms_root + 'test_%03d_target.npz') % self.epoch)

            #################################################
            # For assessing the speaker IDs of SC09 audios, #
            # we treat it as `face recognization` and       #
            # use the triplet loss in FaceNet. You can take #
            # `https://github.com/timesler/facenet-pytorch` #
            # for reference.                                #
            #################################################

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

    args.trace_c = 8
    args.trace_w = 512
    args.nz = 256
    args.lms_w = 128
    if args.dataset == 'SC09':
        args.lms_h = 44
    elif args.dataset == 'Sub-URMP':
        args.lms_h = 22

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

    utils.make_path(args.lms_root)
    utils.make_path(args.ckpt_root)
    utils.make_path(args.audio_root)

    loader = DataLoader(args)

    assert args.dataset == 'SC09'
    train_dataset = SC09Dataset(
                    lms_dir=args.data_path[args.dataset]['media'],
                    npz_dir=args.data_path[args.dataset][args.side],
                    split=args.data_path[args.dataset]['split'][0],
                    trace_c=args.trace_c,
                    trace_w=args.trace_w,
                    max_db=args.max_db,
                    min_db=args.min_db,
                    side=args.side
                )
    args.num_content = train_dataset.content_cnt
    args.num_ID = train_dataset.ID_cnt
    # args.num_content = 10 # SC09
    # args.num_ID = 1477 # SC09
    # args.num_content = 13 # Sub-URMP
    # args.num_ID = 58 # Sub-URMP
    
    test_dataset = SC09Dataset(
                    lms_dir=args.data_path[args.dataset]['media'],
                    npz_dir=args.data_path[args.dataset][args.side],
                    split=args.data_path[args.dataset]['split'][1],
                    trace_c=args.trace_c,
                    trace_w=args.trace_w,
                    max_db=args.max_db,
                    min_db=args.min_db,
                    side=args.side
                )

    engine = AudioEngine(args)

    train_loader = loader.get_loader(train_dataset)
    test_loader = loader.get_loader(test_dataset)

    for i in range(args.num_epoch):
        engine.train(train_loader)
        if i % args.test_freq == 0:
            engine.test(test_loader)
            engine.save_model((args.ckpt_root + '%03d.pth') % (i + 1))
    engine.save_model((args.ckpt_root + 'final.pth'))
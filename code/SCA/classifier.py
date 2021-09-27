import os
import argparse
import numpy as np
import random
import json
import time
import progressbar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms

import utils

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.uniform_(-0.1, 0.1)
        m.bias.data.fill_(0)

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class ClassifierID(nn.Module): # for Sub-URMP
    def __init__(self, n_class, alpha, dim=128, nc=1, dataset='SC09'):
        super(ClassifierID, self).__init__()
        self.alpha = alpha
        self.dim = dim
        nf = 64
        if dataset == 'SC09':
            # input is (nc) x 128 x 44
            self.c1 = nn.Sequential(
                    nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True)
                    )
        else:
            # input is (nc) x 128 x 22
            self.c1 = nn.Sequential(
                    nn.Conv2d(nc, nf, (4, 3), (2, 1), (1, 1), bias=False),
                    nn.LeakyReLU(0.2, inplace=True)
                    )
        # state size. (nf) x 64 x 22
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 11
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 5
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 2
        self.c5 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 4 x 1
        self.c6 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, (4, 3), (1, 1), (0, 1)),
                #nn.Sigmoid()
                )
        self.fc = nn.Sequential(
                    nn.Linear(dim, n_class),
                    nn.Sigmoid()
                )
        self.apply(weights_init)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = self.c6(h5)
        # final feature
        h6 = h6.view(h6.size(0), -1)
        feature = self.l2_norm(h6)
        out = self.fc(feature)
        return feature * self.alpha, out

class ClassifierContent(nn.Module): # for Sub-URMP
    def __init__(self, n_class, alpha, dim=128, nc=1, dataset='SC09'):
        super(ClassifierContent, self).__init__()
        self.alpha = alpha
        self.dim = dim
        nf = 64
        if dataset == 'SC09':
            # input is (nc) x 128 x 44
            self.c1 = nn.Sequential(
                    nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True)
                    )
        else:
            # input is (nc) x 128 x 22
            self.c1 = nn.Sequential(
                    nn.Conv2d(nc, nf, (4, 3), (2, 1), (1, 1), bias=False),
                    nn.LeakyReLU(0.2, inplace=True)
                    )
        # state size. (nf) x 64 x 22
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 11
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 5
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 2
        self.c5 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 4 x 1
        self.c6 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, (4, 3), (1, 1), (0, 1)),
                #nn.Sigmoid()
                )
        self.fc = nn.Sequential(
                    nn.Linear(dim, n_class),
                    nn.Sigmoid()
                )
        self.apply(weights_init)

    def forward(self, x):
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = self.c6(h5)
        h6 = h6.view(h6.size(0), -1)
        out = self.fc(h6)
        return out

class TripletMarginLoss(Function):
    """Triplet loss function.
    """
    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)  # norm 2

    def forward(self, anchor, positive, negative):
        d_p = self.pdist.forward(anchor, positive)
        d_n = self.pdist.forward(anchor, negative)

        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss

class PairwiseDistance(Function):
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        assert x1.size() == x2.size()
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return torch.pow(out + eps, 1. / self.norm)

class TripletDataset(Dataset):
    # def __init__(self, x_seq, y_label, n_triplets):
    #     self.n_triplets = n_triplets
    #     self.triplets_list = self.generate_triplets(x_seq, y_label, self.n_triplets)

    def __init__(self, lms_dir, x_seq, y_label, n_triplets, split='train', max_db=50, min_db=-90):
        self.lms_dir = lms_dir
        self.x_seq = x_seq # filename list
        self.y_label = y_label
        self.n_triplets = n_triplets
        self.norm = transforms.Normalize((0.5,), (0.5,))
        self.max_db = max_db
        self.min_db = min_db
        self.triplets_list = self.generate_triplets(x_seq, y_label, self.n_triplets)

    @staticmethod
    def generate_triplets(x_seq, y_label, num_triplets):
        def create_indices(x_seq, y_label):
            inds = dict()
            for idx, label in enumerate(y_label):
                if label not in inds.keys():
                    inds[label] = []
                inds[label].append(x_seq[idx])
            return inds

        triplets = []
        # Indices = array of labels and each label is an array of indices
        indices = create_indices(x_seq, y_label)

        label_set = set(y_label)
        for i in range(num_triplets):
            c1 = random.sample(label_set, 1)[0]
            c2 = random.sample(label_set, 1)[0]
            while len(indices[c1]) < 2:
                c1 = random.sample(label_set, 1)[0]

            while c1 == c2:
                c2 = random.sample(label_set, 1)[0]
            if len(indices[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1]) - 1)
                n2 = np.random.randint(0, len(indices[c1]) - 1)
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]) - 1)
            if len(indices[c2]) ==1:
                n3 = 0
            else:
                n3 = np.random.randint(0, len(indices[c2]) - 1)

            triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3], c1, c2])
            if i and i % 10000 == 0:
                print('Created %d triplets...' % i)
        return triplets

    def lms_to_tensor(self, lms_name):
        lms = np.load(self.lms_dir + lms_name)
        audio = lms['arr_0']
        audio = audio.astype(np.float32)
        audio = torch.from_numpy(audio).view([1, 128, 22]) # sc09
        audio = utils.my_scale(v=audio,
                               v_max=self.max_db,
                               v_min=self.min_db
                            )
        audio = self.norm(audio)
        return audio

    def __getitem__(self, index):
        a, p, n, c1, c2 = self.triplets_list[index]
        a = self.lms_to_tensor(a)
        p = self.lms_to_tensor(p)
        n = self.lms_to_tensor(n)

        c1 = torch.LongTensor([c1])
        c2 = torch.LongTensor([c2])
        return a, p, n, c1.squeeze(), c2.squeeze()

    def __len__(self):
        return len(self.triplets_list)

class TestDataset(Dataset):
    def __init__(self, recons_dir, target_dir, max_db=50, min_db=-90):
        super(TestDataset).__init__()
        self.recons_dir = recons_dir
        self.target_dir = target_dir
        self.recons_list = sorted(os.listdir(recons_dir))
        self.target_list = sorted(os.listdir(target_dir))
        self.max_db = max_db
        self.min_db = min_db
        self.norm = transforms.Normalize((0.5,), (0.5,))

    def __len__(self):
        return len(self.recons_list)

    def __getitem__(self, index):
        assert self.recons_list[index] == self.target_list[index]
        lms_name = self.recons_list[index]

        recons_lms = np.load(self.recons_dir + lms_name)
        recons_audio = recons_lms['arr_0']
        recons_audio = recons_audio.astype(np.float32)
        recons_audio = torch.from_numpy(recons_audio).view([1, 128, 22]) # sc09
        recons_audio = utils.my_scale(v=recons_audio,
                               v_max=self.max_db,
                               v_min=self.min_db
                            )
        recons_audio = self.norm(recons_audio)

        target_lms = np.load(self.target_dir + lms_name)
        target_audio = target_lms['arr_0']
        target_audio = target_audio.astype(np.float32)
        target_audio = torch.from_numpy(target_audio).view([1, 128, 22]) # sc09
        target_audio = utils.my_scale(v=target_audio,
                               v_max=self.max_db,
                               v_min=self.min_db
                            )
        target_audio = self.norm(target_audio)
        return recons_audio, target_audio, lms_name

class SC09Dataset(Dataset):
    def __init__(self, lms_dir, max_db=50, min_db=-90):
        super(SC09Dataset).__init__()
        self.lms_dir = lms_dir

        self.lms_list = sorted(os.listdir(self.lms_dir))

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

        self.content_list = [] # label list
        self.ID_list = [] # label list
        for file_name in self.lms_list:
            content, ID = file_name.split('_')[:2]
            self.content_list.append(self.content_label[content])
            self.ID_list.append(self.ID_label[ID])
        
        self.max_db = max_db
        self.min_db = min_db
        
        self.norm = transforms.Normalize((0.5,), (0.5,))
        # print('Max db: %f' % self.max_db)
        # print('Min db: %f' % self.min_db)
        #if split == 'train':
        print('N ID: %d' % (len(self.ID_label.keys())))
        print('N Content : %d' % (len(self.content_label.keys())))
        print('N Data Points: %d' % (len(self.lms_list)))

    def __len__(self):
        return len(self.lms_list)

    def __getitem__(self, index):
        lms_name = self.lms_list[index]
        lms = np.load(self.lms_dir + lms_name)
        audio = lms['arr_0']
        audio = audio.astype(np.float32)
        audio = torch.from_numpy(audio).view([1, 128, 44]) # sc09
        audio = utils.my_scale(v=audio,
                               v_max=self.max_db,
                               v_min=self.min_db
                            )
        audio = self.norm(audio)

        content_name, ID_name = lms_name.split('_')[:2]
        content = int(self.content_label[content_name])
        #ID = int(self.ID_label[ID_name])
        content = torch.LongTensor([content])
        #ID = torch.LongTensor([ID])

        return audio, content.squeeze()

class URMPDataset(Dataset):
    def __init__(self, lms_dir, split, max_db=50, min_db=-90):
        super(URMPDataset).__init__()
        self.lms_dir = lms_dir + ('%s/' % split)

        self.lms_list = sorted(os.listdir(self.lms_dir))

        self.content_label = {}
        self.ID_label = {}

        self.content_cnt = 0
        self.ID_cnt = 0

        for file_name in self.lms_list:
            ID = file_name.split('_')[0]
            content = ID[:-2]
            if content not in self.content_label.keys():
                self.content_label[content] = self.content_cnt
                self.content_cnt += 1
            if ID not in self.ID_label.keys():
                self.ID_label[ID] = self.ID_cnt
                self.ID_cnt += 1

        assert self.content_cnt == len(self.content_label.keys())
        assert self.ID_cnt == len(self.ID_label.keys())

        self.content_list = [] # label list
        self.ID_list = [] # label list
        for file_name in self.lms_list:
            ID = file_name.split('_')[0]
            content = ID[:-2]
            self.content_list.append(self.content_label[content])
            self.ID_list.append(self.ID_label[ID])
        
        self.max_db = max_db
        self.min_db = min_db
        
        self.norm = transforms.Normalize((0.5,), (0.5,))
        # print('Max db: %f' % self.max_db)
        # print('Min db: %f' % self.min_db)
        #if split == 'train':
        print('N ID: %d' % (len(self.ID_label.keys())))
        print('N Content: %d' % (len(self.content_label.keys())))
        print('N Data Points: %d' % (len(self.lms_list)))

    def __len__(self):
        return len(self.lms_list)

    def __getitem__(self, index):
        lms_name = self.lms_list[index]
        lms = np.load(self.lms_dir + lms_name)
        audio = lms['arr_0']
        audio = audio.astype(np.float32)
        audio = torch.from_numpy(audio).view([1, 128, 22]) # Sub-URMP
        audio = utils.my_scale(v=audio,
                               v_max=self.max_db,
                               v_min=self.min_db
                            )
        audio = self.norm(audio)

        ID_name = lms_name.split('_')[0]
        content_name = ID_name[:-2]
        content = int(self.content_label[content_name])
        #ID = int(self.ID_label[ID_name])
        content = torch.LongTensor([content])
        #ID = torch.LongTensor([ID])

        return audio, content.squeeze()

class DataLoader(object):
    def __init__(self, args):
        self.args = args
        self.init_param()
        #self.init_dataset()

    def init_param(self):
        self.gpus = torch.cuda.device_count()
        # self.transform = transforms.Compose([
        #                         transforms.Resize(self.args.image_size),
        #                         transforms.ToTensor(),
        #                         transforms.Normalize((0.5,), (0.5,)),
        #                    ])

    def get_loader(self, dataset, shuffle=True):
        data_loader = torch.utils.data.DataLoader(
                            dataset, 
                            batch_size=self.args.batch_size * self.gpus,
                            num_workers=int(self.args.num_workers),
                            shuffle=shuffle
                        )
        return data_loader

class TripletEngine(object):
    def __init__(self, args):
        self.args = args
        self.epoch = 0
        self.ce = nn.CrossEntropyLoss()
        self.l2_dist = PairwiseDistance(2)
        self.similarity = nn.CosineSimilarity(dim=1)
        self.init_model_optimizer()

    def init_model_optimizer(self):
        print('Initializing Model and Optimizer ...')
        self.model = ClassifierID(self.args.num_ID, self.args.alpha, dataset='URMP')
        self.model = torch.nn.DataParallel(self.model, [0]).cuda()
        self.optim = torch.optim.Adam(self.model.module.parameters(), lr=self.args.lr)

    def save_model(self, path):
        state = {
            'model': self.model.module.state_dict()
        }
        torch.save(state, path)
        print('Saving Model on %s ...' % path)

    def load_model(self, path):
        ckpt = torch.load(path)
        self.model.module.load_state_dict(ckpt['model'])
        print('Loading Model from %s ...' % path)

    def get_score(self, distance):
        #print('distance: ', distance.shape) # [bs]
        #print(distance)
        return (1 / (1 + distance)).cpu().data.numpy()

    def is_same(self, score, thresh=0.8):
        correct = (score > thresh)
        #print(score.shape)
        acc = correct.sum() / len(correct)
        return acc

    def train(self, data_loader):
        with torch.autograd.set_detect_anomaly(True):
            self.model.train()
            self.epoch += 1
            record = utils.Record()
            record_cls = utils.Record()
            record_trip = utils.Record()
            acc_cls = utils.Record()
            acc_trip = utils.Record()
            current_time = time.time()
            progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
            for i, (data_a, data_p, data_n, label_p, label_n) in enumerate(data_loader):
                progress.update(i + 1)
                data_a, data_p, data_n = data_a.cuda(), data_p.cuda(), data_n.cuda()
                label_p, label_n = label_p.cuda(), label_n.cuda()
                out_a, _ = self.model(data_a)
                out_p, _ = self.model(data_p)
                out_n, _ = self.model(data_n)

                # choose hard neg
                d_p = self.l2_dist.forward(out_a, out_p)
                d_n = self.l2_dist.forward(out_a, out_n)

                less = (d_p - d_n < 0).cpu().data.numpy().flatten()

                all = (d_n - d_p < self.args.margin).cpu().data.numpy().flatten()
                hard_triplets = np.where(all == 1)
                triplets_index = hard_triplets[0]
                if len(triplets_index):
                    out_selected_a = torch.from_numpy(out_a.cpu().data.numpy()[triplets_index]).cuda()
                    out_selected_p = torch.from_numpy(out_p.cpu().data.numpy()[triplets_index]).cuda()
                    out_selected_n = torch.from_numpy(out_n.cpu().data.numpy()[triplets_index]).cuda()

                    selected_data_a = torch.from_numpy(data_a.cpu().data.numpy()[triplets_index]).cuda()
                    selected_data_p = torch.from_numpy(data_p.cpu().data.numpy()[triplets_index]).cuda()
                    selected_data_n = torch.from_numpy(data_n.cpu().data.numpy()[triplets_index]).cuda()

                    selected_label_p = torch.from_numpy(label_p.cpu().numpy()[triplets_index])
                    selected_label_n= torch.from_numpy(label_n.cpu().numpy()[triplets_index])
                    triplet_loss = TripletMarginLoss(self.args.margin).forward(out_selected_a, out_selected_p, out_selected_n)

                    #print('select', selected_data_a.shape)
                    _, cls_a = self.model(selected_data_a)
                    _, cls_p = self.model(selected_data_p)
                    _, cls_n = self.model(selected_data_n)

                    predicted_labels = torch.cat([cls_a, cls_p, cls_n])
                    true_labels = torch.cat([selected_label_p.cuda(),selected_label_p.cuda(), selected_label_n.cuda()])
                    cross_entropy_loss = self.ce(predicted_labels.cuda(), true_labels.cuda())

                    loss = cross_entropy_loss + triplet_loss

                    record.add(loss.item())
                    record_cls.add(cross_entropy_loss.item())
                    record_trip.add(triplet_loss.item())

                    acc_cls.add(utils.accuracy(predicted_labels, true_labels))
                    acc_trip.add(np.sum(less) / len(less))

                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

            progress.finish()
            print('--------------------')
            print('Epoch :', self.epoch)
            print('Time: %.2f' % (time.time() - current_time), 's')
            print('Loss: %f' % (record.mean()))
            print('Cross Entropy Loss: %f' % ((record_cls.mean())))
            print('Triplet Loss: %f' % ((record_trip.mean())))
            print('Classify Acc: %f' % (acc_cls.mean()))
            print('Triplet Acc: %f' % (acc_trip.mean()))

    def test_triplet(self, data_loader):
        with torch.no_grad():
            self.model.eval()
            record = utils.Record()
            record_cls = utils.Record()
            record_trip = utils.Record()
            acc_cls = utils.Record()
            acc_trip = utils.Record()
            current_time = time.time()
            progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
            for i, (data_a, data_p, data_n, label_p, label_n) in enumerate(data_loader):
                progress.update(i + 1)
                data_a, data_p, data_n = data_a.cuda(), data_p.cuda(), data_n.cuda()
                label_p, label_n = label_p.cuda(), label_n.cuda()
                out_a, _ = self.model(data_a)
                out_p, _ = self.model(data_p)
                out_n, _ = self.model(data_n)

                # choose hard neg
                d_p = self.l2_dist.forward(out_a, out_p)
                d_n = self.l2_dist.forward(out_a, out_n)

                less = (d_p - d_n < 0).cpu().data.numpy().flatten()

                all = (d_n - d_p < self.args.margin).cpu().data.numpy().flatten()
                hard_triplets = np.where(all == 1)
                triplets_index = hard_triplets[0]

                out_selected_a = torch.from_numpy(out_a.cpu().data.numpy()[triplets_index]).cuda()
                out_selected_p = torch.from_numpy(out_p.cpu().data.numpy()[triplets_index]).cuda()
                out_selected_n = torch.from_numpy(out_n.cpu().data.numpy()[triplets_index]).cuda()

                selected_data_a = torch.from_numpy(data_a.cpu().data.numpy()[triplets_index]).cuda()
                selected_data_p = torch.from_numpy(data_p.cpu().data.numpy()[triplets_index]).cuda()
                selected_data_n = torch.from_numpy(data_n.cpu().data.numpy()[triplets_index]).cuda()

                selected_label_p = torch.from_numpy(label_p.cpu().numpy()[triplets_index])
                selected_label_n= torch.from_numpy(label_n.cpu().numpy()[triplets_index])
                triplet_loss = TripletMarginLoss(self.args.margin).forward(out_selected_a, out_selected_p, out_selected_n)

                _, cls_a = self.model(selected_data_a)
                _, cls_p = self.model(selected_data_p)
                _, cls_n = self.model(selected_data_n)

                predicted_labels = torch.cat([cls_a, cls_p, cls_n])
                true_labels = torch.cat([selected_label_p.cuda(),selected_label_p.cuda(), selected_label_n.cuda()])
                cross_entropy_loss = self.ce(predicted_labels.cuda(), true_labels.cuda())

                loss = cross_entropy_loss + triplet_loss

                record.add(loss.item())
                record_cls.add(cross_entropy_loss.item())
                record_trip.add(triplet_loss.item())

                acc_cls.add(utils.accuracy(predicted_labels, true_labels))
                acc_trip.add(np.sum(less) / len(less))
            progress.finish()
            print('--------------------')
            print('Test Triplet:')
            print('Time: %.2f' % (time.time() - current_time), 's')
            print('Loss: %f' % (record.mean()))
            print('Cross Entropy Loss: %f' % ((record_cls.mean())))
            print('Triplet Loss: %f' % ((record_trip.mean())))
            print('Classify Acc: %f' % (acc_cls.mean()))
            print('Triplet Acc: %f' % (acc_trip.mean()))

    def test(self, data_loader, log_path=None):
        with torch.no_grad():
            self.model.eval()
            record = utils.Record()
            log_dict = {}
            current_time = time.time()
            progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
            for i, (recons, target, name_list) in enumerate(data_loader):
                progress.update(i + 1)
                recons_feature, _ = self.model(recons)
                target_feature, _ = self.model(target)
                score = self.get_score(self.l2_dist.forward(recons_feature, target_feature))
                # score = self.similarity(recons_feature, target_feature)
                record.add(self.is_same(score))
                if log_path is not None:
                    for j, name in enumerate(name_list):
                        log_dict[name] = score[j]
            progress.finish()
            if log_path is not None:
                with open(log_path, 'w') as f:
                    json.dump(log_dict, f)
            print('--------------------')
            print('Is Same: %f' % record.mean())
            print('Cost Time: %f' % (time.time() - current_time))


class ClassifierEngine(object):
    def __init__(self, args):
        self.args = args
        self.epoch = 0
        self.ce = nn.CrossEntropyLoss()
        self.init_model_optimizer()

    def init_model_optimizer(self):
        print('Initializing Model and Optimizer ...')
        self.model = ClassifierContent(self.args.num_content, self.args.alpha, dataset='SC09')
        self.model = torch.nn.DataParallel(self.model, [0]).cuda()
        self.optim = torch.optim.Adam(self.model.module.parameters(), lr=self.args.lr)

    def save_model(self, path):
        state = {
            'model': self.model.module.state_dict()
        }
        torch.save(state, path)
        print('Saving Model on %s ...' % path)

    def load_model(self, path):
        ckpt = torch.load(path)
        self.model.module.load_state_dict(ckpt['model'])
        print('Loading Model from %s ...' % path)

    def train(self, data_loader):
        with torch.autograd.set_detect_anomaly(True):
            self.model.train()
            self.epoch += 1
            record = utils.Record()
            record_acc = utils.Record()
            current_time = time.time()
            progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
            for i, (audio, label) in enumerate(data_loader):
                progress.update(i + 1)
                audio = audio.cuda()
                label = label.cuda()
                pred = self.model(audio)
                loss = self.ce(pred, label)
                acc = utils.accuracy(pred, label)
                record.add(loss.item())
                record_acc.add(acc)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            progress.finish()
            print('--------------------')
            print('Epoch :', self.epoch)
            print('Time: %.2f' % (time.time() - current_time), 's')
            print('Loss: %f' % (record.mean()))
            print('Acc: %f' % (record_acc.mean()))

    def test(self, data_loader):
        with torch.no_grad():
            self.model.eval()
            record = utils.Record()
            record_acc = utils.Record()
            current_time = time.time()
            progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
            for i, (audio, label) in enumerate(data_loader):
                progress.update(i + 1)
                audio = audio.cuda()
                label = label.cuda()
                pred = self.model(audio)
                loss = self.ce(pred, label)
                acc = utils.accuracy(pred, label)
                record.add(loss.item())
                record_acc.add(acc)
            progress.finish()
            print('--------------------')
            print('Test')
            print('Time: %.2f' % (time.time() - current_time), 's')
            print('Loss: %f' % (record.mean()))
            print('Acc: %f' % (record_acc.mean()))

if __name__ == '__main__':

    dataset = 'sc09' # ['sc09', 'Sub-URMP']
    op = 'upsample'
    side = 'cacheline'


    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default=('content_classify_%s_upsample' % (dataset)))

    parser.add_argument('--alpha', type=float, default=1, help='weight for final feature')
    parser.add_argument('--margin', type=float, default=0.5, help='margin')

    parser.add_argument('--lms_dir', type=str, default=('/root/data/%s_raw_lms/' % dataset))
    parser.add_argument('--output_root', type=str, default='/root/output/SCA/')

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--beta1', type=float, default=0.5)

    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--test_freq', type=int, default=5)
    parser.add_argument('--max_db', type=float, default=45)
    parser.add_argument('--min_db', type=float, default=-100)
    parser.add_argument('--n_class', type=int, default=-1)
    parser.add_argument('--num_ID', type=int, default=-1)
    parser.add_argument('--num_content', type=int, default=-1)

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

    args.ckpt_root = args.output_root + args.exp_name + '/ckpt/'

    utils.make_path(args.ckpt_root)

    with open(args.output_root + args.exp_name + '/args.json', 'w') as f:
        json.dump(args.__dict__, f)

    loader = DataLoader(args)

    train_audio_dataset = SC09Dataset(
                    lms_dir='',
                    max_db=args.max_db,
                    min_db=args.min_db,
                )
    args.num_content = train_audio_dataset.content_cnt
    args.num_ID = train_audio_dataset.ID_cnt
    
    test_audio_dataset = SC09Dataset(
                    lms_dir='',
                    max_db=args.max_db,
                    min_db=args.min_db,
                )

    # test_recons_dataset = URMPDataset(
    #                 lms_dir=('/root/output/media/%s_%s/' % (dataset, side)),
    #                 split='recons',
    #                 max_db=args.max_db,
    #                 min_db=args.min_db,
    #             )

    train_x_seq = train_audio_dataset.lms_list
    train_ID_label = train_audio_dataset.ID_list

    test_x_seq = test_audio_dataset.lms_list
    test_ID_label = test_audio_dataset.ID_list

    train_triplet_dataset = TripletDataset(
                    lms_dir=args.lms_dir + 'train/',
                    split='train',
                    x_seq=train_x_seq,
                    y_label=train_ID_label,
                    n_triplets=250000,
                    max_db=args.max_db,
                    min_db=args.min_db
                )

    test_triplet_dataset = TripletDataset(
                    lms_dir=args.lms_dir + 'validation/',
                    split='test',
                    x_seq=test_x_seq,
                    y_label=test_ID_label,
                    n_triplets=100000,
                    max_db=args.max_db,
                    min_db=args.min_db
                )

    test_dataset = TestDataset(
                    recons_dir=('/root/output/media/%s_%s/recons/' % (dataset, side)),
                    target_dir=('/root/output/media/%s_%s/target/' % (dataset, side)),
                    max_db=args.max_db,
                    min_db=args.min_db
                )

    # engine = TripletEngine(args)

    # train_loader = loader.get_loader(train_triplet_dataset)
    # test_loader = loader.get_loader(test_dataset)

    # engine.load_model((args.ckpt_root + '%03d.pth') % 6)
    # engine.test(test_loader)
    # engine.test_triplet(loader.get_loader(test_triplet_dataset))

    # engine = ClassifierEngine(args)

    # test_recons_loader = loader.get_loader(test_recons_dataset)
    # engine.load_model((args.ckpt_root + '%03d.pth') % 26)

    # engine.test(test_recons_loader)

    # train_loader = loader.get_loader(train_audio_dataset)
    # test_loader = loader.get_loader(test_audio_dataset)

    # for i in range(args.num_epoch):
    #     engine.train(train_loader)
    #     if (i + 0) % args.test_freq == 0:
    #         engine.test(test_loader)
    #         engine.save_model((args.ckpt_root + '%03d.pth') % (i + 1))
    # model.save_model((args.ckpt_root + 'final.pth'))
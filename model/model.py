import math
import os
import os.path as osp
import random
import sys

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tordata

from .network import TripletLoss, SetNet  # 三元组损失，SetNet
from .utils import TripletSampler, evaluation, evaluation_re_ranking, evaluation_gallery
from tqdm import tqdm
from common import print_log
import time

sample_type = 'all'
frame_num = 30


def my_collate_fn(batch):
    # i为第几个 train=4*4 ,test=1
    # frame*64*44
    # batch[i][0][0] (101, 64, 44) 第i个视频帧数
    # batch[i][1] (101,) 第i个视频帧数
    # batch[i][2] view
    # batch[i][3] 视频序列号
    # batch[i][4] person
    batch_size = len(batch)
    feature_num = len(batch[0][0])
    seqs = [batch[i][0] for i in range(batch_size)]  # batch[0][0]
    frame_sets = [batch[i][1] for i in range(batch_size)]
    view = [batch[i][2] for i in range(batch_size)]
    seq_type = [batch[i][3] for i in range(batch_size)]
    label = [batch[i][4] for i in range(batch_size)]

    batch = [seqs, view, seq_type, label, None]

    # batch [5][i]

    def select_frame(index):
        sample = seqs[index]
        frame_set = frame_sets[index]
        if sample_type == 'random':
            frame_id_list = random.choices(frame_set, k=frame_num)
            _ = [feature.loc[frame_id_list].values for feature in sample]
        else:
            _ = [feature.values for feature in sample]
        return _

    seqs = list(map(select_frame, range(len(seqs))))  # select_frame -> (30,64,44)

    if sample_type == 'random':
        seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
    else:  # 全采样
        gpu_num = 1
        batch_per_gpu = math.ceil(batch_size / gpu_num)
        batch_frames = [[
            len(frame_sets[i])
            for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
            if i < batch_size
        ] for _ in range(gpu_num)]  # 全采样时每一个batch的每一个seq的帧数
        if len(batch_frames[-1]) != batch_per_gpu:
            for _ in range(batch_per_gpu - len(batch_frames[-1])):
                batch_frames[-1].append(0)
        max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])  # 每一个batch总帧数
        seqs = [[
            np.concatenate([
                seqs[i][j]
                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                if i < batch_size
            ], 0) for _ in range(gpu_num)]
            for j in range(feature_num)]
        seqs = [np.asarray([
            np.pad(seqs[j][_],
                   ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                   'constant',
                   constant_values=0)
            for _ in range(gpu_num)])
            for j in range(feature_num)]
        #
        batch[4] = np.asarray(batch_frames)

    batch[0] = seqs

    # batch[0][0] (1,1273,64,44) 一个batch所有帧 (30采样)加起来等于4*4*30
    # batch[1][i] view
    # batch[2][i] 视频动作序列号
    # batch[3][i] person
    # batch[4][i] 视频帧数 batch加起来(全采样)等于1273 (30采样)加起来等于4*4*30
    return batch


class Model:
    def __init__(self,
                 hidden_dim,
                 lr,
                 hard_or_full_trip,
                 batch_size,
                 restore_iter,
                 total_iter,
                 margin,
                 num_workers,
                 frame_num,
                 model_name,
                 save_path,
                 data_source,  # 数据集dataset
                 test_load_iter):

        # self.device = "cpu" # "cuda" or "cpu"
        # self.device_ids = [0,1,2]

        self.device = "cuda"  # "cuda" or "cpu"
        self.device_ids = [0]

        self.save_path = save_path
        self.data_source = data_source
        self.test_load_iter = test_load_iter

        self.hidden_dim = hidden_dim
        self.lr = lr
        self.hard_or_full_trip = hard_or_full_trip  # 'full'
        self.margin = margin
        self.frame_num = frame_num  # 每个视频提取30帧
        self.num_workers = num_workers
        self.batch_size = batch_size  # (4, 4)
        self.model_name = model_name  # 'GaitSet'
        self.P, self.M = batch_size  # 4,4

        self.restore_iter = restore_iter
        self.total_iter = total_iter

        self.encoder = SetNet(self.hidden_dim).float()
        if self.device == "cuda":
            self.encoder = nn.DataParallel(self.encoder, device_ids=self.device_ids)
        self.encoder.to(self.device)

        self.triplet_loss = TripletLoss(self.margin).float()
        if self.device == "cuda":
            self.triplet_loss = nn.DataParallel(self.triplet_loss, device_ids=self.device_ids)
        self.triplet_loss.to(self.device)

        self.optimizer = optim.Adam([
            {'params': self.encoder.parameters()},
        ], lr=self.lr)

        self.hard_loss = []
        self.full_loss = []
        self.loss_num = []
        self.mean_dist = []

        self.sample_type = 'all'

        self.time = time.time()

    def collate_fn(self, batch):
        batch_size = len(batch)
        feature_num = len(batch[0][0])
        seqs = [batch[i][0] for i in range(batch_size)]
        frame_sets = [batch[i][1] for i in range(batch_size)]
        view = [batch[i][2] for i in range(batch_size)]
        seq_type = [batch[i][3] for i in range(batch_size)]
        label = [batch[i][4] for i in range(batch_size)]
        batch = [seqs, view, seq_type, label, None]

        def select_frame(index):
            sample = seqs[index]
            frame_set = frame_sets[index]
            if self.sample_type == 'random':
                frame_id_list = random.choices(frame_set, k=self.frame_num)
                _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                _ = [feature.values for feature in sample]
            return _

        seqs = list(map(select_frame, range(len(seqs))))

        if self.sample_type == 'random':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
        else:
            gpu_num = 1
            batch_per_gpu = math.ceil(batch_size / gpu_num)
            batch_frames = [[
                len(frame_sets[i])
                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                if i < batch_size
            ] for _ in range(gpu_num)]
            if len(batch_frames[-1]) != batch_per_gpu:
                for _ in range(batch_per_gpu - len(batch_frames[-1])):
                    batch_frames[-1].append(0)
            max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])
            seqs = [[
                np.concatenate([
                    seqs[i][j]
                    for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                    if i < batch_size
                ], 0) for _ in range(gpu_num)]
                for j in range(feature_num)]
            seqs = [np.asarray([
                np.pad(seqs[j][_],
                       ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                       'constant',
                       constant_values=0)
                for _ in range(gpu_num)])
                for j in range(feature_num)]
            batch[4] = np.asarray(batch_frames)

        batch[0] = seqs

        return batch

    def run_train(self):
        if self.restore_iter != 0:
            self.load(self.restore_iter)

        self.encoder.train()
        self.sample_type = 'random'

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        triplet_sampler = TripletSampler(self.data_source, self.batch_size)  # 抽样实例

        train_loader = tordata.DataLoader(
            dataset=self.data_source,  # 当 __getitem__ 返回这些数据后，DataLoader 会调用它4*4次（根据采样器的结果index）以构建一个 batch
            batch_sampler=triplet_sampler,  # 自定义从数据集中取样本的策略，但是一次只返回一个batch的indices（索引）一个batch有4*4个index
            collate_fn=my_collate_fn,  # 处理dataset得到的batch
            num_workers=self.num_workers)

        train_label_set = list(self.data_source.label_set)
        train_label_set.sort()
        #
        # import pdb
        # pdb.set_trace()

        self.restore_iter = int(self.restore_iter)
        for data in tqdm(train_loader, total=self.total_iter - self.restore_iter):
            seq, _, _, label, batch_frame = data
            self.restore_iter += 1
            self.optimizer.zero_grad()

            for i in range(len(seq)):
                # 一个batch的总帧数图片
                seq[i] = self.np2var(seq[i]).float()
            if batch_frame is not None:
                # batch_frame (1,16) 每一个视频的帧数
                batch_frame = self.np2var(batch_frame).int()

            # feature (16,128,62) 62的由来，两个31维度的特征concat成62维度 ,128为隐藏层数目
            feature, _ = self.encoder(*seq, batch_frame)

            # target_label对应的person的id
            target_label = [train_label_set.index(l) for l in label]
            target_label = self.np2var(np.array(target_label)).long()

            full_loss, hard_loss, mean_dist, loss_num = self.triplet_loss(feature, target_label)
            if self.hard_or_full_trip == 'hard':
                loss = hard_loss.mean()
            elif self.hard_or_full_trip == 'full':
                loss = full_loss.mean()

            self.hard_loss.append(hard_loss.mean().data.cpu().numpy())  # 难样本度量损失
            self.full_loss.append(full_loss.mean().data.cpu().numpy())  # 全样本度量损失
            self.loss_num.append(loss_num.mean().data.cpu().numpy())
            self.mean_dist.append(mean_dist.mean().data.cpu().numpy())

            if loss > 1e-9:  # 如果loss大于阈值，反向传播Adam优化
                loss.backward()
                self.optimizer.step()

            if self.restore_iter % 1000 == 0:  # 每训练1000次，保存一次模型
                self.save(self.restore_iter)

            if self.restore_iter % 100 == 0:
                now = time.time()
                mess = 'Iteration {0:0>5}, Cost {1:.2f}s, '.format(self.restore_iter, now - self.time)
                mess += 'triplet_loss={0:.4f}, triplet_hard_loss={1:.4f}, triplet_loss_num={2:.4f}, triplet_mean_dist={3:.4f}'.format(
                    np.mean(self.full_loss), np.mean(self.hard_loss), np.mean(self.loss_num), np.mean(self.mean_dist))
                print_log(mess)
                sys.stdout.flush()
                self.hard_loss = []
                self.full_loss = []
                self.loss_num = []
                self.mean_dist = []

                self.time = time.time()

            if self.restore_iter == self.total_iter:
                break

    def ts2var(self, x):
        return autograd.Variable(x).to(self.device)

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))

    def run_test(self, batch_size=1):
        self.load(self.restore_iter)
        self.encoder.eval()

        source = self.data_source
        self.sample_type = 'all'

        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,  # 每次取一张图片
            sampler=tordata.sampler.SequentialSampler(source),  # 它会根据数据集的索引（从 0 到 len(source) - 1）依次返回index
            collate_fn=my_collate_fn,
            num_workers=self.num_workers)

        feature_list = list()
        views_list = list()
        types_list = list()
        label_list = list()

        for x in tqdm(data_loader):
            # import pdb
            # pdb.set_trace()
            # seq (101, 64, 44)
            seq, view, seq_type, label, batch_frame = x
            for j in range(len(seq)):
                seq[j] = self.np2var(seq[j]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()
            # 预测向量[1, 128, 62]
            feature, _ = self.encoder(*seq, batch_frame)

            feature_list.append(feature.data.cpu().numpy())
            views_list += view
            types_list += seq_type
            label_list += label

        # 最后形成每个人每一个动作对应的向量表
        data = (np.concatenate(feature_list, 0), views_list, types_list, label_list)
        evaluation(data, 'CASIA-B', 'euc')
        # evaluation_re_ranking(data, 'CASIA-B', 'euc')
        # evaluation_gallery(data, 'CASIA-B', 'euc')

    def save(self, iteration):
        os.makedirs(self.save_path, exist_ok=True)
        checkpoint = {
            'model': self.encoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iteration': iteration}
        save_name = osp.join(self.save_path, '{}-{:0>5}.pt'.format(self.model_name, iteration))
        torch.save(checkpoint, save_name)

    def load(self, iteration):
        save_name = osp.join(self.save_path, '{}-{:0>5}.pt'.format(self.model_name, iteration))
        checkpoint = torch.load(save_name)
        # import pdb
        # pdb.set_trace()
        self.encoder.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

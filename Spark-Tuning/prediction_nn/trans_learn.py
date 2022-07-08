#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: trans_learn.py 
@create: 2021/6/7 15:21 
"""

import time
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.autograd import Variable
from config import Config
from my_dataset import CodeDagDataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dis(x)
        return x


def load_data(set_type):
    if set_type == 'train':
        s = ''
    else:
        s = 'test/'
    X, CODE, Y, NODES, ADJS = np.load('dataset_' + machine_num + '/' + s + 'X.npy'), np.load(
        'dataset_' + machine_num + '/' + s + 'CODE.npy'), np.load('dataset_' + machine_num + '/' + s + 'Y.npy'), \
                              np.load('dataset_' + machine_num + '/' + s + 'NODES.npy', allow_pickle=True), np.load(
        'dataset_' + machine_num + '/' + s + 'ADJS.npy', allow_pickle=True)
    X, CODE, Y, NODES, ADJS = torch.from_numpy(X).float(), torch.from_numpy(CODE).long(), torch.from_numpy(Y).float(), \
                              torch.from_numpy(NODES).float(), torch.from_numpy(ADJS)
    return X, CODE, Y, NODES, ADJS


if __name__ == '__main__':
    config = Config()
    # discriminator
    D = Discriminator(config.hidden_dim)
    D = D.cuda()
    # generator
    model = torch.load("model_save_1/cnn_gcn_5.pt")
    G = model
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.00002)
    scaler = torch.load("model_save_1/scaler.pt")
    machine_num = '1'
    X, CODE, Y, NODES, ADJS = load_data('train')
    X_test, CODE_test, Y_test, NODES_test, ADJS_test = load_data('test')
    X = scaler.transform(X)
    X_test = scaler.transform(X_test)
    train_dataset = CodeDagDataset(X, CODE, Y, NODES, ADJS)
    test_dataset = CodeDagDataset(X_test, CODE_test, Y_test, NODES_test, ADJS_test)
    loss_weight = 10000
    for i in range(100):
        start = time.time()
        train_indices = random.choices([i for i in range(len(train_dataset))], k=config.trans_batch_size)
        test_indices = random.choices([i for i in range(len(test_dataset))], k=config.trans_batch_size)
        x, code, y, nodes, adj = train_dataset[train_indices]
        x = torch.from_numpy(x)
        x_test, code_test, y_test, nodes_test, adj_test = test_dataset[test_indices]
        x_test = torch.from_numpy(x_test)
        if torch.cuda.is_available():
            x, code, y, nodes, adj = x.cuda(), code.cuda(), y.cuda(), nodes.cuda(), adj.cuda()
            x_test, code_test, y_test, nodes_test, adj_test = x_test.cuda(), code_test.cuda(), \
                                                              y_test.cuda(), nodes_test.cuda(), adj_test.cuda()
        # true -> in train set
        true_label = Variable(torch.ones(config.trans_batch_size)).cuda()
        false_label = Variable(torch.zeros(config.trans_batch_size)).cuda()
        # train discriminator to distinguish correctly
        mixed_feature = G.get_mix_feature(x.float(), code.long(), nodes.float(), adj.float())
        label = D(mixed_feature)
        d_loss1 = bce_loss(label.squeeze(-1), true_label)
        mixed_feature = G.get_mix_feature(x_test.float(), code_test.long(), nodes_test.float(), adj_test.float())
        label = D(mixed_feature)
        d_loss2 = bce_loss(label.squeeze(-1), false_label)
        d_loss = d_loss1 + d_loss2
        print(str(i) + " d_loss: " + str(d_loss.item()))
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        # train generator to make test sample hard to distinguish
        mixed_feature = G.get_mix_feature(x.float(), code.long(), nodes.float(), adj.float())
        label = D(mixed_feature)
        y_pred = G(x.float(), code.long(), nodes.float(), adj.float())
        g_loss_1 = loss_weight * bce_loss(label.squeeze(-1), false_label) + torch.sqrt(mse_loss(y_pred, y))
        mixed_feature = G.get_mix_feature(x_test.float(), code_test.long(), nodes_test.float(), adj_test.float())
        label = D(mixed_feature)
        y_pred = G(x.float(), code.long(), nodes.float(), adj.float())
        g_loss_2 = loss_weight * bce_loss(label.squeeze(-1), true_label) + torch.sqrt(mse_loss(y_pred, y_test))
        g_loss = g_loss_1 + g_loss_2
        print(str(i) + " g_loss: " + str(g_loss.item()))
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        torch.save(G, 'model_save_trans_1/trans' + str(i) + '.pt')
        print('********************************************************')




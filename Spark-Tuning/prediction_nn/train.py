#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: train.py 
@create: 2021/5/21 20:57 
"""

import time
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from models import *
from my_dataset import CodeDagDataset
from config import Config
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    config = Config()
    machine_num = '1'
    X, CODE, Y, NODES, ADJS = np.load('dataset_'+machine_num+'/X.npy'), np.load('dataset_'+machine_num+'/CODE.npy'), np.load('dataset_'+machine_num+'/Y.npy'), \
                       np.load('dataset_'+machine_num+'/NODES.npy', allow_pickle=True), np.load('dataset_'+machine_num+'/ADJS.npy', allow_pickle=True)
    scaler = MinMaxScaler()
    scaler.fit(X)
    torch.save(scaler, 'model_save_'+machine_num+'/scaler.pt')
    X = scaler.transform(X)
    # model = BiLstmText(config.code_vocab_size, config.code_emb_dim,
    #                    hidden_dim=config.code_hidden_dim, n_layers=config.code_lstm_layers)
    # model = textCNN(config.code_vocab_size, config.code_emb_dim, config.code_hidden_dim)
    model = MultiModel(config.code_vocab_size, config.code_emb_dim, config.code_hidden_dim)
    # model = AllCNN(config.code_vocab_size, config.code_emb_dim, config.code_hidden_dim)
    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    # dataset = MyDataset(X, CODE, Y)
    dataset = CodeDagDataset(X, CODE, Y, NODES, ADJS)
    train_set, validate_set = random_split(dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset)) + 1])
    train_loader = DataLoader(dataset=train_set, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=validate_set, batch_size=config.batch_size, shuffle=True)

    for epoch in range(config.epoch_num):
        if (epoch+1) % config.lr_decay_epochs == 0:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * config.lr_decay_ratio
        model.train()
        train_all_y = []
        train_all_pred = []
        for i, batch in enumerate(train_loader):
            start = time.time()
            #print(str(epoch) + ": " + str(i))
            tensor_x, tensor_code, tensor_y, nodes, adj = batch
            if torch.cuda.is_available():
                tensor_x, tensor_code, tensor_y = tensor_x.cuda(), tensor_code.cuda(), tensor_y.cuda()
                nodes, adj = nodes.cuda(), adj.cuda()
            # y_pred = model.forward(tensor_x.float(), tensor_code.long())
            y_pred = model.forward(tensor_x.float(), tensor_code.long(), nodes.float(), adj.float())
            loss = criterion(y_pred, tensor_y.float().unsqueeze(1))
            # print('     loss: ' + str(loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # train metrics
            predicted = y_pred.cpu().data
            train_all_pred.extend(predicted.numpy().tolist())
            train_all_y.extend(tensor_y.cpu().long().numpy())
            # break
        train_mae = mean_absolute_error(train_all_y, np.array(train_all_pred).flatten())
        train_rmse = np.sqrt(mean_squared_error(train_all_y, np.array(train_all_pred).flatten()))
        print("epoch " + str(epoch) + ": train MAE: " + str(train_mae) + '; RMSE: ' + str(train_rmse))
        torch.save(model, 'model_save_'+machine_num+'/cnn_gcn_' + str(epoch) + '.pt')
        model.eval()
        valid_all_y = []
        valid_all_pred = []
        for i, batch in enumerate(valid_loader):
            tensor_x, tensor_code, tensor_y, nodes, adj = batch
            if torch.cuda.is_available():
                tensor_x, tensor_code, tensor_y = tensor_x.cuda(), tensor_code.cuda(), tensor_y.cuda()
                nodes, adj = nodes.cuda(), adj.cuda()
            # y_pred = model.forward(tensor_x.float(), tensor_code.long())
            y_pred = model.forward(tensor_x.float(), tensor_code.long(), nodes.float(), adj.float())
            predicted = y_pred.cpu().data
            valid_all_pred.extend(predicted.numpy().tolist())
            valid_all_y.extend(tensor_y.cpu().long().numpy())
        val_mae = mean_absolute_error(valid_all_y, np.array(valid_all_pred).flatten())
        val_rmse = np.sqrt(mean_squared_error(valid_all_y, np.array(valid_all_pred).flatten()))
        print("epoch " + str(epoch) + ": valid MAE: " + str(val_mae) + '; RMSE: ' + str(val_rmse))
        end = time.time()
        print("time cost: " + str(end-start))
        print("#####################################################################")
    print('end')


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: models.py 
@create: 2020/12/21 15:19 
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class AllCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(AllCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # code convs
        out_channels = 64
        kernel_sizes = [3, 4, 5]
        self.code_convs = nn.ModuleList([nn.Conv2d(1, out_channels, (k, hidden_dim)) for k in kernel_sizes])
        # DAG node conv
        self.node_convs = nn.ModuleList([nn.Conv1d(1, out_channels, k) for k in kernel_sizes])
        # DAG adj conv
        self.adj_convs = nn.ModuleList([nn.Conv1d(64, out_channels, k) for k in kernel_sizes])
        self.dag_code_fc = nn.Sequential(
            nn.Linear(3 * out_channels * len(kernel_sizes), hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 29, hidden_dim), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1), nn.ReLU()
        )

    def compute_code(self, code):
        emb = self.embedding(code)  # (batch_size, seq_len, emb_dim)
        emb = emb.unsqueeze(1)  # (batch_size, 1, seq_len, emb_dim)  (64, 1, 1000, 128)
        code_out = [F.relu(conv(emb)).squeeze(3) for conv in
                    self.code_convs]  # len(kernel_sizes) * (batch_size, k_num, seq_len-k)  3*(64,64,(1000-k))
        code_out = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in
                    code_out]  # len(kernel_sizes)*(batch_size, k_num)  3*(64,64)
        code_out = torch.cat(code_out, 1)  # (batch_size, k_num*len(kernel_sizes))  (64, 64*3)
        return code_out

    def compute_nodes(self, nodes):
        nodes = nodes.unsqueeze(1)  # [batch_size, 1, node_num]
        nodes_out = [F.relu(conv(nodes)) for conv in
                     self.node_convs]  # len(kernel_sizes) * (batch_size, k_num, seq_len-k)  3*(64,64,(1000-k))
        nodes_out = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in
                     nodes_out]  # len(kernel_sizes)*(batch_size, k_num)  3*(64,64)
        nodes_out = torch.cat(nodes_out, 1)  # (batch_size, k_num*len(kernel_sizes))  (64, 64*3)
        return nodes_out

    def compute_adj(self, adj):
        adj_out = [F.relu(conv(adj)) for conv in
                   self.adj_convs]  # len(kernel_sizes) * (batch_size, k_num, seq_len-k)  3*(64,64,(1000-k))
        adj_out = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in
                   adj_out]  # len(kernel_sizes)*(batch_size, k_num)  3*(64,64)
        adj_out = torch.cat(adj_out, 1)  # (batch_size, k_num*len(kernel_sizes))  (64, 64*3)
        return adj_out

    def forward(self, x, code, nodes, adj):
        code_out = self.compute_code(code)
        nodes_out = self.compute_nodes(nodes)
        adj_out = self.compute_adj(adj)
        out = torch.cat((code_out, nodes_out, adj_out), dim=1)
        out = self.dag_code_fc(out)
        cat = torch.cat((out, x), dim=1)
        return self.mlp(cat)


class MultiModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(MultiModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # code convs
        k_num = 64
        kernel_sizes = [3, 4, 5]
        self.convs = nn.ModuleList([nn.Conv2d(1, k_num, (k, hidden_dim)) for k in kernel_sizes])
        self.dropout = nn.Dropout(0.1)
        # DAG gcn
        self.gcn = GCN(1, 16)  # node feature数量为1，期望输出
        # DAG cnn
        self.mlp = nn.Sequential(
            nn.Linear(len(kernel_sizes) * k_num + 64 + 29, hidden_dim), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1), nn.ReLU(),
            #nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1), nn.ReLU(), # new
            #nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1), nn.ReLU(), # new
            #nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1), nn.ReLU(), # new
            #nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1), nn.ReLU(), # new
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1), nn.ReLU()
        )

    def forward(self, x, code, nodes, adj):
        emb = self.embedding(code)  # (batch_size, seq_len, emb_dim)
        emb = emb.unsqueeze(1)  # (batch_size, 1, seq_len, emb_dim)  (64, 1, 1000, 128)
        cnn_out = [F.relu(conv(emb)).squeeze(3) for conv in
                   self.convs]  # len(kernel_sizes) * (batch_size, k_num, seq_len-k)  3*(64,64,(1000-k))
        cnn_out = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in
                   cnn_out]  # len(kernel_sizes)*(batch_size, k_num)  3*(64,64)
        cnn_out = torch.cat(cnn_out, 1)  # (batch_size, k_num*len(kernel_sizes))  (64, 64*3)
        cnn_out = self.dropout(cnn_out)
        nodes = nodes.unsqueeze(-1)
        gcn_out = self.gcn(nodes, adj)
        gcn_out = gcn_out.squeeze(-1)
        cat = torch.cat((gcn_out, cnn_out, x), dim=1)
        return self.mlp(cat)


class textCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(textCNN, self).__init__()
        k_num = 64
        kernel_sizes = [3, 4, 5]

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.convs = nn.ModuleList([nn.Conv2d(1, k_num, (k, hidden_dim)) for k in kernel_sizes])
        self.dropout = nn.Dropout(0.1)
        # self.fc = nn.Linear(len(kernel_sizes) * k_num, Cla)
        self.mlp = nn.Sequential(
            nn.Linear(len(kernel_sizes) * k_num + 29, 256), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(256, 256), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(256, 256), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(256, 128), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(128, 64), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(64, 1), nn.ReLU()
        )

    def forward(self, x, code):
        emb = self.embedding(code)  # (N,W,D)
        emb = emb.unsqueeze(1)  # (N,Ci,W,D)
        cnn_out = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # len(Ks)*(N,Knum,W)
        cnn_out = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in cnn_out]  # len(Ks)*(N,Knum)

        cnn_out = torch.cat(cnn_out, 1)
        cnn_out = self.dropout(cnn_out)
        cat = torch.cat((cnn_out, x), dim=1)
        return self.mlp(cat)


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, init='xavier'):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(input_dim, hidden_dim, init=init)
        self.gc2 = GraphConvolution(hidden_dim, 1, init=init)
        self.dropout = 0.1

    def bottleneck(self, path1, path2, path3, adj, in_x):
        return F.relu(path3(F.relu(path2(F.relu(path1(in_x, adj)), adj)), adj))

    def forward(self, nodes, adj):
        x = F.dropout(F.relu(self.gc1(nodes, adj)), self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
        # return F.log_softmax(x, dim=1)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, init='xavier'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if init == 'uniform':
            print("| Uniform Initialization")
            self.reset_parameters_uniform()
        elif init == 'xavier':
            print("| Xavier Initialization")
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            print("| Kaiming Initialization")
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02)  # Implement Xavier Uniform

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')

    def forward(self, nodes, adj):
        # nodes [batch_size, node_num, node_feat_num]
        # weight [node_feat_num, output_dim]
        # support [batch_size, node_num, output_dim]
        support = torch.matmul(nodes, self.weight)
        # adj [batch_size, node_num, node_num]
        # output = [torch.spmm(adj[i], support[i]) for i in range(nodes.shape[0])]
        # output = torch.stack(output, dim=0)
        output = torch.matmul(adj, support)
        return output


class BiLstmText(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers):
        super(BiLstmText, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.5)
        self.mlp = nn.Sequential(
            nn.Linear(285, 128), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(128, 64), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(64, 1), nn.ReLU()
        )

    # x,query：[batch, seq_len, hidden_dim*2]
    def attention_net(self, x, query, mask=None):  # 软性注意力机制（key=value=x）
        d_k = query.size(-1)  # d_k为query的维度
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  # 打分机制  scores:[batch, seq_len, seq_len]
        p_attn = F.softmax(scores, dim=-1)  # 对最后一个维度归一化得分
        context = torch.matmul(p_attn, x).sum(1)  # 对权重化的x求和，[batch, seq_len, hidden_dim*2]->[batch, hidden_dim*2]
        return context, p_attn

    def forward(self, x, code):
        embedding = self.dropout(self.embedding(code))  # [seq_len, batch, embedding_dim]
        # output: [seq_len, batch, hidden_dim*2]     hidden/cell: [n_layers*2, batch, hidden_dim]
        output, (final_hidden_state, final_cell_state) = self.rnn(embedding)
        output = output.permute(1, 0, 2)  # [batch, seq_len, hidden_dim*2]

        query = self.dropout(output)
        attn_output, attention = self.attention_net(output, query)
        cat = torch.cat((attn_output, x), dim=1)
        # logit = self.fc(attn_output)
        return self.mlp(cat)

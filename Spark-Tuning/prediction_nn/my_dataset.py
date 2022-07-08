#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: my_dataset.py 
@create: 2020/12/21 16:46 
"""

import torch
import numpy as np
from torch.utils.data import Dataset


class TransDataset(Dataset):
    def __init__(self, feat, code, y, dag_nodes, dag_adjs, label):
        self.feat = feat
        self.code = code
        self.y = y
        self.dag_nodes = dag_nodes
        self.dag_adjs = dag_adjs
        self.label = label

    def __getitem__(self, index):
        feat, code, y = self.feat[index], self.code[index], self.y[index]
        dag_node, dag_adj = self.dag_nodes[index], self.dag_adjs[index]
        label = self.label[index]
        return feat, code, y, dag_node, dag_adj, label

    def __len__(self):
        return len(self.y)


class CodeDagDataset(Dataset):
    def __init__(self, feat, code, y, dag_nodes, dag_adjs):
        self.feat = feat
        self.code = code
        self.y = y
        self.dag_nodes = dag_nodes
        self.dag_adjs = dag_adjs

    def __getitem__(self, index):
        feat, code, y = self.feat[index], self.code[index], self.y[index]
        dag_node, dag_adj = self.dag_nodes[index], self.dag_adjs[index]
        return feat, code, y, dag_node, dag_adj

    def __len__(self):
        return len(self.y)


class MyDataset(Dataset):
    def __init__(self, x, code, y):
        self.x = x
        self.code = code
        self.y = y

    def __getitem__(self, index):
        x, code, y = self.x[index], self.code[index], self.y[index]
        return x, code, y

    def __len__(self):
        return len(self.y)


#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: dataset_process.py 
@create: 2021/5/21 19:47 
"""

import torch
import numpy as np
import pandas as pd
from data_process_text import load_all_code_ids

workload_dict = {
    "Spark ConnectedComponent Application": "CC",
    "DecisionTree classification Example": "DT",
    "Spark KMeans Example": "KM",
    "LinerRegressionApp Example": "LiR",
    "LogisticRegressionApp Example": "LoR",
    "Spark LabelPropagation Application": "LP",
    "MFApp Example": "MF",
    "Spark PCA Example": "PCA",
    "Spark PregelOperation Application": "PO",
    "Spark PageRank Application": "PR",
    "Spark StronglyConnectedComponent Application": "SCC",
    "Spark ShortestPath Application": "SP",
    "Spark SVDPlusPlus Application": "SVD",
    "SVM Classifier Example": "SVM",
    "Spark TriangleCount Application": "TC",
    "TeraSort": "TS"
}


def read_stage_dataset(dataset_path):
    ws2nodes = torch.load('dag_data/ws2nodes.pth')
    ws2adj = torch.load('dag_data/ws2adj.pth')
    all_code_dict = load_all_code_ids()
    df = pd.read_csv(dataset_path, sep=',', low_memory=False)
    workload, stage_id = df['AppName'].apply(lambda x: workload_dict[x]), df['stage_id']
    code_vec = []
    nodes_vec = []
    adj_vec = []
    W, S = [], []
    for w, s in zip(workload, stage_id):
        W.append(w)
        S.append(s)
        try:
            code_vec.append(all_code_dict[w][str(s)])
        except:
            code_vec.append([0 for _ in range(1000)])
        try:
            nodes_vec.append(ws2nodes[w][str(s)])
            adj_vec.append(ws2adj[w][str(s)])
        except:
            nodes_vec.append(np.zeros(64, dtype=float))
            adj_vec.append(np.zeros([64, 64], dtype=float))
    Y = df['duration']
    X_df = df.drop(['AppId', 'AppName', 'Duration', 'code', 'duration', 'stage_id'], axis=1)
    X = X_df.values
    return X, code_vec, Y, W, S, nodes_vec, adj_vec


if __name__ == '__main__':
    machine_num = '3'
    X, CODE, Y, W, S, NODES, ADJS = read_stage_dataset(dataset_path='test_data/dataset_test_'+machine_num+'.csv')
    print(X.shape)
    print(len(CODE))
    print(Y.shape)
    print(len(NODES))
    print(len(ADJS))
    print(len(W))
    print(len(S))
    np.save('dataset_'+machine_num+'/test/X.npy', X)
    np.save('dataset_'+machine_num+'/test/CODE.npy', CODE)
    np.save('dataset_'+machine_num+'/test/Y.npy', Y)
    np.save('dataset_'+machine_num+'/test/NODES.npy', NODES)
    np.save('dataset_'+machine_num+'/test/ADJS.npy', ADJS)
    np.save('dataset_'+machine_num+'/test/W.npy', W)
    np.save('dataset_'+machine_num+'/test/S.npy', S)


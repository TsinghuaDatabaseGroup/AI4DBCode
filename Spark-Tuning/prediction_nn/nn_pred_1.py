#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: nn_pred.py 
@create: 2021/5/22 18:42 
"""
import os
import random
import numpy as np
import pandas as pd
import torch
from evaluation import eval_regression
from check_order import eval_ranking
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


def eval_workload_all(df, w_name):
    data_rows = df.head(1)['rows'].tolist()[0]
    groups = df.groupby('AppId')
    all_target = []
    all_pred = []
    i = 0
    for app_id, group in groups:
        if group.head(1)['rows'].tolist()[0] != data_rows:
            continue
        total_y = group['Duration'].tolist()[0]
        workload, stage_id = group['AppName'].apply(lambda x: workload_dict[x]), group['stage_id']
        code_vec = []
        nodes_vec = []
        adj_vec = []
        for w, s in zip(workload, stage_id):
            try:
                code_vec.append(all_code_dict[w][str(s)])
                nodes_vec.append(ws2nodes[w][str(s)])
                adj_vec.append(ws2adj[w][str(s)])
            except:
                code_vec.append([0 for _ in range(1000)])
                nodes_vec.append(np.zeros(64, dtype=float))
                adj_vec.append(np.zeros([64, 64], dtype=float))
        CODE = np.array(code_vec)
        nodes_vec = np.stack(nodes_vec)
        adj_vec = np.stack(adj_vec)
        Y = group['duration']
        X = group.drop(['AppId', 'AppName', 'Duration', 'code', 'duration', 'stage_id'], axis=1).values
        # workload_target = Y.sum()
        workload_target = total_y
        X = scaler.transform(X)
        X = torch.FloatTensor(X)
        CODE = torch.LongTensor(CODE)
        NODES = torch.FloatTensor(nodes_vec)
        ADJS = torch.FloatTensor(adj_vec)
        if torch.cuda.is_available():
            X = X.cuda()
            CODE = CODE.cuda()
        try:
            workload_pred = model(X, CODE, NODES, ADJS)
        except:
            continue
        workload_pred = workload_pred.cpu().sum()
        all_target.append(workload_target)
        all_pred.append(workload_pred)
        i += 1
        if i == 100:
            break
    res = evl(np.array(all_target), np.array(all_pred))
    return res


def evl(all_target, all_pred):
    times = 10
    res1, res2 = np.array([0.0 for _ in range(5)]), np.array([0.0 for _ in range(3)])
    for _ in range(times):
        sample_indices = random.choices([i for i in range(all_pred.shape[0])], k=25)
        sample_pred = all_pred[sample_indices]
        sample_target = all_target[sample_indices]
        r1, r2 = eval_regression(sample_target, sample_pred), \
                 eval_ranking(sample_target, sample_pred)
        res1 += r1
        res2 += r2
    return res1 / times, res2 / times


def main(dataset_path):
    eval_result, ranking_result = [], []
    df_all = pd.read_csv(dataset_path, sep=',', low_memory=False)
    df_workloads = df_all.groupby('AppName')
    for w_name, w_df in df_workloads:
        res, ranking_res = eval_workload_all(w_df, w_name)
        eval_result.append(res)
        ranking_result.append(ranking_res)
    print("regression eval: ")
    print(np.mean(np.array(eval_result), axis=0))
    print("ranking eval: ")
    print(np.mean(np.array(ranking_result), axis=0))
    return np.mean(np.array(eval_result), axis=0)[3]


if __name__ == '__main__':
    random.seed(2021)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    all_code_dict = load_all_code_ids()
    ws2nodes = torch.load('dag_data/ws2nodes.pth')
    ws2adj = torch.load('dag_data/ws2adj.pth')
    scaler = torch.load('model_save_1/scaler.pt')
    data_path = 'test_data/dataset_test_1.csv'
    tuples = []
    for i in range(50, 100):
        if torch.cuda.is_available():
            model = torch.load('model_save_1/cnn_gcn_'+str(i)+'.pt')
        else:
            #model = torch.load('model_save_1/cnn_gcn_'+str(i)+'.pt', map_location='cpu')
            model = torch.load('model_save_trans_1/trans'+str(i)+'.pt', map_location='cpu')
        hr = main(data_path)
        t = (hr, i)
        tuples.append(t)
    tuples.sort(key=lambda x: x[0])
    tuples.reverse()
    for i in range(5):
        print('hr: ' + str(tuples[i][0]) + 'model: ' + str(tuples[i][1]))





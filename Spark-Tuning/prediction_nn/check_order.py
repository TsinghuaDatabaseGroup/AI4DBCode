#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: check_order.py 
@create: 2021/3/25 18:08 
"""

import random
import numpy as np
import pandas as pd
from sklearn.metrics import label_ranking_average_precision_score, coverage_error, label_ranking_loss


def eval_ranking(y_target, y_pred):
    y_workload, y_stage_added = np.array(y_target), np.array(y_pred)
    stacked = np.stack((y_workload, y_stage_added), axis=1)
    stacked = stacked[np.lexsort(stacked[:, ::-1].T)]
    weighted_indices = []
    for i in range(y_workload.shape[0]):
        if i < 5:
            weighted_indices.extend([i]*10)
        elif i < 20:
            weighted_indices.extend([i]*5)
        else:
            weighted_indices.append(i)
    y_true, y_score = [], []
    for _ in range(5000):
        # i = random.randint(0, y_workload.shape[0])
        # j = random.randint(0, y_workload.shape[0])
        i = random.choice(weighted_indices)
        j = random.choice(weighted_indices)
        if y_workload[i] <= y_workload[j]:
            y_true.append([0, 1])
        else:
            y_true.append([1, 0])
        y_score.append([y_stage_added[i], y_stage_added[j]])
    y_true, y_score = np.array(y_true), np.array(y_score)
    lrap = label_ranking_average_precision_score(y_true, y_score)
    #print(lrap)
    cov_err = coverage_error(y_true, y_score)
    ranking_loss = label_ranking_loss(y_true, y_score)
    return np.array([lrap, cov_err, ranking_loss])


def check_a_workload(workload_name, df):
    groups = df.groupby('AppId')
    y_stage_added = []
    y_workload = []
    for name, group in groups:
        y_workload.append(group['Duration'].tolist()[0])
        Y = group['duration']
        y_stage_added.append(Y.sum())
    return eval_ranking(y_workload, y_stage_added)


if __name__ == '__main__':
    # dataset_path = 'dataset_by_stage_merged/dataset_by_stage.csv'
    dataset_path = 'dataset_by_stage_merged/dataset_test.csv'
    eval_result = []
    df_all = pd.read_csv(dataset_path, sep=',', low_memory=False)
    df_workloads = df_all.groupby('AppName')
    all_res = []
    for w_name, w_df in df_workloads:
        all_res.append(check_a_workload(w_name, w_df))
    print(np.mean(np.array(all_res), axis=0))

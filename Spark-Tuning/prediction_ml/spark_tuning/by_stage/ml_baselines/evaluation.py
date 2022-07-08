#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: evaluation.py 
@create: 2020/10/9 17:02 
"""

import math
import numpy as np
from numpy import mean
import heapq
from sklearn.metrics import mean_squared_error, mean_absolute_error


def eval_regression_(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('mae: ', mae)
    print('rmse: ', rmse)
    return np.array([mae, rmse])


def eval_regression(y_test, y_pred, isUseTopk=True, k=5):
    y_test = np.array(y_test) / 1000
    y_pred = np.array(y_pred) / 1000
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('mae: ', round(mae, 4), 'rmse: ', round(rmse, 4))
    if not isUseTopk:
        return np.array([mae, rmse])
    #print(y_pred[:16])
    map_pred_score = {}
    map_test_score = {}
    for i in range(y_pred.shape[0]):
        map_pred_score[i] = y_pred[i]
        map_test_score[i] = y_test[i]
    pred_ranklist = heapq.nsmallest(k, map_pred_score, key=map_pred_score.get)
    top_k = heapq.nsmallest(k, map_test_score, key=map_test_score.get)
    top_k2 = heapq.nsmallest(2*k, map_test_score, key=map_test_score.get)
    hr = getHitRatio(pred_ranklist[:k], top_k)
    ndcg = getNDCG(pred_ranklist[:k], top_k, top_k2)
    mrr = getMRR(pred_ranklist[:k], top_k)
    #print(HRs, NDCGs, MRRs)
    print('HR = ', round(hr, 4), ', NDCG = ', round(ndcg, 4), ', MRR = ', round(mrr, 4))
    return np.array([mae, rmse, hr, ndcg, mrr])


def getHitRatio(ranklist, top_k):
    hit_count = 0
    for item in ranklist:
        if item in top_k:
            hit_count += 1
    return hit_count / len(ranklist)


def getNDCG(ranklist, top_k, top_k2):
    s = 0
    real_s = 0
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item in top_k:
            real_s += 1 / math.log(i+2, 2)
        if item in top_k2:
            real_s += 1 / math.log(i+2, 2)
        s += 2 / math.log(i+2, 2)
    return real_s / s


def getMRR(ranklist, top_k):
    s = 0
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item in top_k:
            s += 1/(ranklist.index(item)+1)
    return s / len(ranklist)

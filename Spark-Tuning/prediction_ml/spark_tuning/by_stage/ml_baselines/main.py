#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: main.py 
@create: 2020/10/9 17:52 
"""
from by_stage.ml_baselines.models import *
from sklearn.model_selection import train_test_split


def run_model(model):
    X, Y = np.load('npy_dataset_all/X_tfidf_emb_1.npy'), np.load('npy_dataset_all/Y_emb_1.npy')
    # X, Y = np.load('npy_dataset_all/X_tfidf_emb_8.npy'), np.load('npy_dataset_all/Y_emb_8.npy')
    eval_all = []
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
        eval = model(X_train, X_test, y_train, y_test, times=i)
        eval_all.append(eval)
    print(np.mean(np.array(eval_all), axis=0))
    print('end')


if __name__ == '__main__':
    # svr_regression  bayesian_ridge_regression  linear_regression  sgd_regression
    # lasso_regression  gbr_regression  mlp_regression rf_regression
    run_model(light_gbm_cv)

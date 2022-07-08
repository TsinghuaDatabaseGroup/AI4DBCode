#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: main.py 
@create: 2020/10/9 17:52 
"""
from by_workload.ml_baselines.data_process_new import *
from by_workload.ml_baselines.models import *
from sklearn.model_selection import train_test_split


def shuffle(X, Y):
    state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(state)
    np.random.shuffle(Y)


def run_model(model):
    X, Y = read_merged_data("./merged_file/merged_dataset.csv")
    eval_all = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
        eval = model(X_train, X_test, y_train, y_test)
        eval_all.append(eval)
    print(np.mean(np.array(eval_all), axis=0))
    print('end')


def run_model_specify_testset(model):
    X_train, y_train = read_merged_data('merged_file/dataset_train.csv', use_tf_idf=False)
    X_test, y_test = read_merged_data('merged_file/dataset_test.csv', use_tf_idf=False)

    eval_all = []
    for i in range(10):
        shuffle(X_train, y_train)
        eval = model(X_train, X_test, y_train, y_test)
        eval_all.append(eval)
    print(np.mean(np.array(eval_all), axis=0))
    print('end')


if __name__ == '__main__':
    # svr_regression  bayesian_ridge_regression  linear_regression  sgd_regression
    # lasso_regression  gbr_regression  mlp_regression rf_regression light_gbm
    # run_model(mlp_regression)
    run_model_specify_testset(mlp_regression)
    print()

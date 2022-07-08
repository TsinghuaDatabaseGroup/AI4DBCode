#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: xgboost_rank.py 
@create: 2021/5/7 18:29 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import DMatrix, train
from sklearn.externals import joblib
from by_stage.ml_baselines.data_process_all import get_code_vec, workload_dict
from by_stage.ml_baselines.evaluation import *
from by_stage.build_dataset.check_order import eval_ranking


def example():
    xgb_rank_params1 = {
        'booster': 'gbtree',
        'eta': 0.1,
        'gamma': 1.0,
        'min_child_weight': 0.1,
        'objective': 'rank:pairwise',
        'eval_metric': 'merror',
        'max_depth': 6,
        'num_boost_round': 10,
        'save_period': 0
    }
    xgb_rank_params2 = {
        'bst:max_depth': 2,
        'bst:eta': 1, 'silent': 1,
        'objective': 'rank:pairwise',
        'nthread': 4,
        'eval_metric': 'ndcg'
    }

    # generate training dataset
    # 一共2组*每组3条，6条样本，特征维数是2
    n_group = 2
    n_dim = 3
    dtrain = np.random.uniform(0, 100, [n_group * n_dim, 2])
    # numpy.random.choice(a, size=None, replace=True, p=None)
    dtarget = np.array([np.random.choice([0, 1, 2], 3, False) for i in range(n_group)]).flatten()
    # n_group用于表示从前到后每组各自有多少样本，前提是样本中各组是连续的，[3，3]表示一共6条样本中前3条是第一组，后3条是第二组
    dgroup = np.array([n_dim for _ in range(n_group)]).flatten()

    # concate Train data, very import here !
    xgbTrain = DMatrix(dtrain, label=dtarget)
    xgbTrain.set_group(dgroup)

    # generate eval data
    dtrain_eval = np.random.uniform(0, 100, [n_group * n_dim, 2])
    xgbTrain_eval = DMatrix(dtrain_eval, label=dtarget)
    xgbTrain_eval.set_group(dgroup)
    evallist = [(xgbTrain, 'train'), (xgbTrain_eval, 'eval')]

    # train model
    # xgb_rank_params1加上 evals 这个参数会报错，还没找到原因
    # rankModel = train(xgb_rank_params1,xgbTrain,num_boost_round=10)
    rankModel = train(xgb_rank_params2, xgbTrain, num_boost_round=20, evals=evallist)

    # test dataset
    dtest = np.random.uniform(0, 100, [n_group * n_dim, 2])
    dtestgroup = np.array([n_dim for _ in range(n_group)]).flatten()
    xgbTest = DMatrix(dtest)
    xgbTest.set_group(dgroup)

    # test
    print(rankModel.predict(xgbTest))


def read_data(data_type, from_file=True):
    if from_file:
        X = joblib.load('tmp_file/by_stage/' + data_type + '_X.pickle')
        Y = joblib.load('tmp_file/by_stage/' + data_type + '_Y.pickle')
        app_id_all = joblib.load('tmp_file/by_stage/' + data_type + '_app_id_all.pickle')
        app_name_all = joblib.load('tmp_file/by_stage/' + data_type + '_app_name_all.pickle')
        workload_Y = joblib.load('tmp_file/by_stage/' + data_type + '_Y_w.pickle')
        return X, Y, app_id_all, app_name_all, workload_Y
    if data_type == 'train':
        dataset_path = '../ml_baselines/csv_dataset/dataset_by_stage.csv'
    else:
        dataset_path = '../ml_baselines/test_data/dataset_test.csv'
    df = pd.read_csv(dataset_path, sep=',', low_memory=False)
    df = df.sort_values(by='AppName')
    # workload + stage id -> TF-IDF / n-gram
    workload, stage_id = df['AppName'].apply(lambda x: workload_dict[x]), df['stage_id']
    tf_idf_vec, tf_idf_labels = get_code_vec(workload, stage_id, "tfidf", 128)
    code_vec = np.array(tf_idf_vec)
    Y = df['duration']
    app_id_all = df['AppId']
    app_name_all = df['AppName']
    workload_Y = df['Duration']
    X_df = df.drop(['AppId', 'AppName', 'Duration', 'code', 'duration'], axis=1)
    X = X_df.values[:, :]
    code_vec = code_vec.squeeze(axis=1)
    X = np.concatenate((X, code_vec), axis=1)
    joblib.dump(X, 'tmp_file/by_stage/' + data_type + '_X.pickle')
    joblib.dump(Y, 'tmp_file/by_stage/' + data_type + '_Y.pickle')
    joblib.dump(app_id_all, 'tmp_file/by_stage/' + data_type + '_app_id_all.pickle')
    joblib.dump(app_name_all, 'tmp_file/by_stage/' + data_type + '_app_name_all.pickle')
    joblib.dump(workload_Y, 'tmp_file/by_stage/' + data_type + '_Y_w.pickle')
    return X, Y, app_id_all, app_name_all, workload_Y


def create_dataset(X, Y, app_name_all):
    # 分为几组，每组各有多少样本
    counts = app_name_all.value_counts()
    n_group = len(counts)
    dgroup = [0 for _ in range(n_group)]
    app_name_list = app_name_all.tolist()
    idx = 0
    dgroup[0] = 1
    for i in range(1, len(app_name_list)):
        if i > 0 and app_name_list[i] != app_name_list[i - 1]:
            idx += 1
        dgroup[idx] += 1
    # 封装训练数据
    dataset = DMatrix(X, label=Y)
    dataset.set_group(dgroup)
    return dataset


def ranking_train():
    X_train, Y_train, app_ids_train, app_names_train, Y_train_w = read_data('train', from_file=True)
    train_data = create_dataset(X_train, Y_train, app_names_train)
    X_test, Y_test, app_ids_test, app_names_test, Y_test_w = read_data('test', from_file=True)
    test_data = create_dataset(X_test, Y_test, app_names_test)

    # 训练
    xgb_rank_params = {
        'booster': 'gbtree',
        'n_estimators': 200,
        'max_depth': 10,
        'eta': 1,
        'silent': 1,
        'objective': 'rank:pairwise',
        'nthread': 20,
        'eval_metric': 'ndcg'
    }
    xgb_regression_params = {
        'booster': 'gbtree',
        'n_estimators': 200,
        'max_depth': 10,
        'eta': 1,
        'silent': 1,
        'objective': 'reg:squarederror',
        'nthread': 20,
        'eval_metric': 'rmse'
    }
    eval_list = [(train_data, 'train'), (test_data, 'eval')]
    rank_model = train(xgb_rank_params, train_data, num_boost_round=20, evals=eval_list)
    joblib.dump(rank_model, 'saved_model/xgboost_rank.pickle')


def ranking_test():
    rank_model = joblib.load('saved_model/xgboost_rank.pickle')
    X_test, Y_test, app_ids_test, app_names_test, Y_test_w = read_data('test', from_file=True)
    test_data = create_dataset(X_test, Y_test, app_names_test)
    result = rank_model.predict(test_data)
    pred = pd.Series(result)
    pred.name = 'pred'
    df_all = pd.concat([app_ids_test, app_names_test, pred, Y_test, Y_test_w], axis=1)
    df_workloads = df_all.groupby('AppName')
    eval_result, ranking_result = [], []
    for w_name, w_df in df_workloads:
        print(w_name)
        res, ranking_res = eval_a_workload(w_df)
        eval_result.append(res)
        ranking_result.append(ranking_res)
    print("regression eval: ")
    print(np.mean(np.array(eval_result), axis=0))
    print("ranking eval: ")
    print(np.mean(np.array(ranking_result), axis=0))
    return


def eval_a_workload(df):
    groups = df.groupby('AppId')
    all_target = []
    all_pred = []
    for app_id, group in groups:
        total_y = group['Duration'].tolist()[0]
        Y = group['duration']
        # 两种目标值：累加、总时间
        # workload_target = Y.sum()
        workload_target = total_y
        pred = group['pred'].sum()
        all_target.append(workload_target)
        all_pred.append(pred)
    res = observe(all_target, all_pred)
    return res


def observe(all_target, all_pred):
    best_target_idx = heapq.nsmallest(5, range(len(all_target)), all_target.__getitem__)
    # best_pred_idx = set(map(all_pred.index, heapq.nsmallest(10, all_pred)))
    best_pred_idx = heapq.nsmallest(5, range(len(all_pred)), all_pred.__getitem__)
    # best_target = heapq.nsmallest(10, all_target)
    # best_pred = heapq.nsmallest(10, all_pred)
    best_pred = [all_target[i] for i in best_pred_idx]
    best_target = [all_target[i] for i in best_target_idx]
    all_target, all_pred = np.array(all_target), np.array(all_pred)
    stacked = np.stack((all_target, all_pred), axis=1)
    plot(stacked)
    return eval_regression(all_target, all_pred), eval_ranking(all_target, all_pred)


def plot(stacked_data):
    stacked_data = stacked_data[np.lexsort(stacked_data[:, ::-1].T)]
    # 设置绘图风格
    plt.style.use('ggplot')
    # 设置中文编码和负号的正常显示
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot([i for i in range(stacked_data.shape[0])],  # x轴数据
             stacked_data[:, 0],  # y轴数据
             linestyle='-',  # 折线类型
             linewidth=2,  # 折线宽度
             color='steelblue',  # 折线颜色
             marker='o',  # 点的形状
             markersize=6,  # 点的大小
             markeredgecolor='black',  # 点的边框色
             markerfacecolor='brown',
             label='target')  # 点的填充色
    plt.plot([i for i in range(stacked_data.shape[0])],  # x轴数据
             stacked_data[:, 1],  # y轴数据
             linestyle='-',  # 折线类型
             linewidth=2,  # 折线宽度
             color='#ff9999',  # 折线颜色
             marker='o',  # 点的形状
             markersize=6,  # 点的大小
             markeredgecolor='black',  # 点的边框色
             markerfacecolor='#ff9999',  # 点的填充色
             label='pred')  # 添加标签
    plt.title('target VS pred')
    plt.xlabel('range')
    plt.ylabel('duration')
    plt.legend()
    plt.show()


def main():
    ranking_train()
    ranking_test()


if __name__ == '__main__':
    # example()
    main()

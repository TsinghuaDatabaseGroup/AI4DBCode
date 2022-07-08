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
import random
import numpy as np
import matplotlib.pyplot as plt
from xgboost import DMatrix, train
from sklearn.externals import joblib
from by_stage.ml_baselines.data_process_all import get_code_vec, workload_dict
from by_stage.ml_baselines.evaluation import *
from by_stage.build_dataset.check_order import eval_ranking


def process_data(data_type):
    if data_type == 'train':
        dataset_path = '../ml_baselines/csv_dataset/dataset_by_stage_1.csv'
    else:
        dataset_path = '../ml_baselines/test_data/dataset_test_8.csv'
    df = pd.read_csv(dataset_path, sep=',', low_memory=False)
    app_names = []
    final_df = pd.DataFrame()
    dgroup = []
    df_workloads = df.groupby('AppName')
    i = 1
    for w_name, w_df in df_workloads:
        print(w_name)
        workload, stage_id = w_df['AppName'].apply(lambda x: workload_dict[x]), w_df['stage_id']
        tf_idf_vec, tf_idf_labels = get_code_vec(workload, stage_id, 'tfidf', 128)
        code_vec = np.array(tf_idf_vec)
        w_df = w_df.drop(['code', 'stage_id'], axis=1)
        w_df.index = range(len(w_df))
        code_vec = code_vec.squeeze(axis=1)
        code_df = pd.DataFrame(code_vec)
        w_df = pd.concat([w_df, code_df], axis=1)
        agg_df = w_df.groupby('AppId').mean()
        app_names.extend([w_name for _ in range(len(agg_df))])
        dgroup.append(len(agg_df))
        final_df = final_df.append(agg_df)
        i += 1
    print(dgroup)
    joblib.dump(app_names, 'tmp_file/' + data_type + '_app_names.pickle')
    final_df.to_csv('tmp_file/' + data_type + '_rank_data.csv', sep=',', encoding='utf-8')
    joblib.dump(dgroup, 'tmp_file/' + data_type + '_rank_dgroup.pickle')


def load_data(data_type):
    app_names = joblib.load('tmp_file/' + data_type + '_app_names.pickle')
    dgroup = joblib.load('tmp_file/' + data_type + '_rank_dgroup.pickle')
    df = pd.read_csv('tmp_file/' + data_type + '_rank_data.csv', sep=',', low_memory=False)
    Y = df['Duration']
    X = df.drop(['Duration', 'AppId'], axis=1)
    dataset = DMatrix(X, label=Y)
    dataset.set_group(dgroup)
    return dataset, app_names, Y


def ranking_train():
    train_data, _, _ = load_data('train')
    test_data, _, _ = load_data('test')

    # 训练
    xgb_rank_params = {
        'booster': 'gbtree',
        'max_depth': 50,
        'eta': 1,
        'gamma': 0.1,
        'min_child_weight': 0.01,
        'silent': 1,
        'objective': 'rank:pairwise',
        'nthread': 20,
        'num_boost_round': 10,
        'eval_metric': 'ndcg'
    }
    eval_list = [(train_data, 'train'), (test_data, 'eval')]
    rank_model = train(xgb_rank_params, train_data, num_boost_round=20, evals=eval_list)
    joblib.dump(rank_model, 'saved_model/xgboost_rank_workload.pickle')


def ranking_test():
    rank_model = joblib.load('saved_model/xgboost_rank_workload.pickle')
    test_data, test_app_names, Y_test = load_data('test')
    result = rank_model.predict(test_data)
    pred = pd.Series(result)
    pred.name = 'pred'
    test_app_names = pd.Series(test_app_names)
    test_app_names.name = 'AppName'
    df_all = pd.concat([test_app_names, pred, Y_test], axis=1)
    df_workloads = df_all.groupby('AppName')
    eval_result, ranking_result = [], []
    for w_name, w_df in df_workloads:
        print(w_name)
        # all evaluation
        # res, ranking_res = eval_regression(w_df['Duration'], w_df['pred']), \
        #                    eval_ranking(w_df['Duration'], w_df['pred'])
        # sample evaluation
        all_pred = np.array(w_df['pred'])
        all_target = np.array(w_df['Duration'])
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
        eval_result.append(res1/times)
        ranking_result.append(res2/times)
    print("regression eval: ")
    print(np.mean(np.array(eval_result), axis=0))
    print("ranking eval: ")
    print(np.mean(np.array(ranking_result), axis=0))
    return


def main():
    ranking_train()
    ranking_test()


if __name__ == '__main__':
    # process_data('train')
    # process_data('test')

    main()

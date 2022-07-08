#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: predict_workload.py 
@create: 2021/1/8 10:09 
"""

from sklearn.externals import joblib
from by_stage.ml_baselines.data_process_all import *
from by_stage.ml_baselines.evaluation import *
from by_stage.build_dataset.check_order import eval_ranking


def eval_workload_one_line(df):
    groups = df.groupby('AppId')
    all_target = []
    all_pred = []
    for name, group in groups:
        # workload + stage id -> TF-IDF / n-gram
        workload, stage_id = group['AppName'].apply(lambda x: workload_dict[x]), group['stage_id']
        tf_idf_vec = get_code_vec(workload, stage_id, 'tfidf', 128)
        code_vec = np.array(tf_idf_vec)
        total_y = group['Duration'].tolist()[0]
        Y = group['duration']
        X_df = group.drop(['AppId', 'AppName', 'Duration', 'code', 'duration'], axis=1)
        X = X_df.values[:, 0:]
        code_vec = code_vec.squeeze(axis=1)
        X = np.concatenate((X, code_vec), axis=1)
        # workload_target = Y.sum()
        workload_target = total_y
        workload_pred = model.predict(X)
        workload_pred = workload_pred.sum()
        all_target.append(workload_target)
        all_pred.append(workload_pred)
    res = observe(all_target, all_pred)
    return res


def eval_workload_all(df):
    groups = df.groupby('AppId')
    all_target = []
    all_pred = []
    i = 0
    for app_id, group in groups:
        # workload + stage id -> TF-IDF / n-gram
        workload, stage_id = group['AppName'].apply(lambda x: workload_dict[x]), group['stage_id']
        tf_idf_vec, tf_idf_labels = get_code_vec(workload, stage_id, 'tfidf', 128)
        code_vec = np.array(tf_idf_vec)
        total_y = group['Duration'].tolist()[0]
        Y = group['duration']
        X_df = group.drop(['AppId', 'AppName', 'Duration', 'code', 'duration'], axis=1)
        X_labels = X_df.columns
        X_labels = np.concatenate((X_labels, tf_idf_labels))
        X = X_df.values[:, 0:]
        code_vec = code_vec.squeeze(axis=1)
        X = np.concatenate((X, code_vec), axis=1)
        # 目标值累加
        workload_target = Y.sum()
        # workload_target = total_y
        workload_pred = model.predict(X)
        # 特征重要性
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        for f in range(X.shape[1]):
            print(X_labels[indices[f]] + " = " + str(importance[indices[f]]))
        workload_pred = workload_pred.sum()
        all_target.append(workload_target)
        all_pred.append(workload_pred)
        # print(str(i) + ': ' + app_id)
        i += 1
        if i == 100:
            break
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
    return eval_regression(all_target, all_pred), eval_ranking(all_target, all_pred)


def main():
    # dataset_path = 'test_data/dataset_test.csv'
    dataset_path = 'csv_dataset/dataset_by_stage.csv'
    eval_result, ranking_result = [], []
    df_all = pd.read_csv(dataset_path, sep=',', low_memory=False)
    df_workloads = df_all.groupby('AppName')
    for w_name, w_df in df_workloads:
        res, ranking_res = eval_workload_all(w_df)
        eval_result.append(res)
        ranking_result.append(ranking_res)
    print("regression eval: ")
    print(np.mean(np.array(eval_result), axis=0))
    print("ranking eval: ")
    print(np.mean(np.array(ranking_result), axis=0))

    # stacked = np.load('test_data/output.npy')
    # all_target, all_pred = stacked[0], stacked[1]
    # observe(all_target, all_pred)


if __name__ == '__main__':
    model = joblib.load('saved_model/gbr_all_2.pickle')
    main()

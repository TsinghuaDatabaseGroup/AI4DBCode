#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: pred_by_workload.py 
@create: 2021/3/25 19:28 
"""

import random
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from by_workload.ml_baselines.data_process_new import *
from by_workload.ml_baselines.evaluation import *
from by_stage.build_dataset.check_order import eval_ranking


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


def eval_workload_level(name, df, file):
    X, Y = df2npy(df, use_tf_idf=use_tf_idf)
    # X, Y = df2npy(df, use_tf_idf=False, use_embedding=True)
    # X, Y = df2npy(df, use_tf_idf=False, use_embedding=False, use_inst_code=True)
    if scaling:
        X = scaler.transform(X)
    y_pred = model.predict(X)
    if scaling:
        y_pred = y_pred * 100
    file.write(workload_dict[name]+': \n')
    file.write('target: ' + ', '.join([str(i) for i in Y.tolist()]) + '\n')
    file.write('predict: ' + ', '.join([str(int(i)) for i in y_pred.tolist()]) + '\n')
    return observe(Y, y_pred)


def observe(all_target, all_pred):
    best_target_idx = heapq.nsmallest(len(all_target), range(len(all_target)), all_target.__getitem__)
    # best_pred_idx = set(map(all_pred.index, heapq.nsmallest(10, all_pred)))
    best_pred_idx = heapq.nsmallest(len(all_target), range(len(all_pred)), all_pred.__getitem__)
    # best_target = heapq.nsmallest(10, all_target)
    # best_pred = heapq.nsmallest(10, all_pred)
    best_pred = [all_target[i] for i in best_pred_idx]
    best_target = [all_target[i] for i in best_target_idx]
    best_target, best_pred = np.array(best_target), np.array(best_pred)
    best_stacked = np.stack((best_target, best_pred), axis=1)
    # **********
    all_target, all_pred = np.array(all_target), np.array(all_pred)
    stacked = np.stack((all_target, all_pred), axis=1)
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


def main():
    eval_result, ranking_result = [], []
    df_all = pd.read_csv(dataset_path, sep=',', low_memory=False)
    df_workloads = df_all.groupby('AppName')
    for w_name, w_df in df_workloads:
        w_df = w_df.head(100)
        print(w_name)
        file = open('tmp_output/output_'+workload_dict[w_name], 'w', encoding='utf-8')
        res, ranking_res = eval_workload_level(w_name, w_df, file)
        eval_result.append(res)
        ranking_result.append(ranking_res)
    print("regression eval: ")
    print(np.mean(np.array(eval_result), axis=0))
    print("ranking eval: ")
    print(np.mean(np.array(ranking_result), axis=0))


scaling = False   # SVR

if __name__ == '__main__':
    dataset_path = 'merged_file/dataset_test_8.csv'
    scaler = joblib.load('saved_model/svr_scaler.pickle')
    use_tf_idf = False
    if use_tf_idf:
        model = joblib.load('saved_model/gbm_13_1.pickle')
    else:
        model = joblib.load('saved_model/mlp_no_code_8.pickle')
    main()












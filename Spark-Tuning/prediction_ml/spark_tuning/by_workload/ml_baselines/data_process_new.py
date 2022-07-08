#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: data_process_new.py 
@create: 2021/3/25 17:34 
"""

import os
import re
import numpy as np
import json
import csv
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib


all_fields = ['AppId', 'AppName', 'Duration',
              'spark.default.parallelism', 'spark.driver.cores', 'spark.driver.memory', 'spark.driver.maxResultSize', 'spark.executor.instances', 'spark.executor.cores', 'spark.executor.memory', 'spark.executor.memoryOverhead', 'spark.files.maxPartitionBytes', 'spark.memory.fraction', 'spark.memory.storageFraction', 'spark.reducer.maxSizeInFlight', 'spark.shuffle.compress', 'spark.shuffle.file.buffer', 'spark.shuffle.spill.compress',
              'rows', 'cols', 'itr', 'partitions',
              'node_num', 'cpu_cores', 'cpu_freq', 'mem_size', 'mem_speed', 'net_width']

env_1_1000M = {"node_num": 1, "cpu_cores": 16, "cpu_freq": 3.2, "mem_size": 64, "mem_speed": 2400, "net_width": 1000}
env_3_1000M = {"node_num": 3, "cpu_cores": 16, "cpu_freq": 3.2, "mem_size": 64, "mem_speed": 2400, "net_width": 1000}
env_8_1000M = {"node_num": 8, "cpu_cores": 16, "cpu_freq": 2.9, "mem_size": 16, "mem_speed": 2666, "net_width": 1000}
env_8_10000M = {"node_num": 8, "cpu_cores": 16, "cpu_freq": 2.9, "mem_size": 16, "mem_speed": 2666, "net_width": 10000}

envs = {
    "1_1000M": env_1_1000M, "3_1000M": env_3_1000M, "8_1000M": env_8_1000M, "8_10000M": env_8_10000M
}


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


workload2file = {
    "Spark ConnectedComponent Application": "tf-idf/ConnectedComponentApp.npy",
    "DecisionTree classification Example": "tf-idf/DecisionTreeApp.npy",
    "Spark KMeans Example": "tf-idf/kmeans.npy",
    "LinerRegressionApp Example": "tf-idf/LinearRegressionApp.npy",
    "LogisticRegressionApp Example": "tf-idf/LogisticRegressionApp.npy",
    "Spark LabelPropagation Application": "tf-idf/LabelPropagationApp.npy",
    "MFApp Example": "tf-idf/MFMovieLens.npy",
    "Spark PCA Example": "tf-idf/PCAApp.npy",
    "Spark PregelOperation Application": "tf-idf/PregelOperationApp.npy",
    "Spark PageRank Application": "tf-idf/pagerankApp.npy",
    "Spark StronglyConnectedComponent Application": "tf-idf/StronglyConnectedComponentApp.npy",
    "Spark ShortestPath Application": "tf-idf/ShortestPathsApp.npy",
    "Spark SVDPlusPlus Application": "tf-idf/SVDPlusPlusApp.npy",
    "SVM Classifier Example": "tf-idf/DocToTFIDF.npy",
    "Spark TriangleCount Application": "tf-idf/triangleCountApp.npy",
    "TeraSort": "tf-idf/terasortApp.npy"
}

workload2emb = {
    "Spark ConnectedComponent Application": "dag_emb/CC_embedding",
    "DecisionTree classification Example": "dag_emb/DT_embedding",
    "Spark KMeans Example": "dag_emb/KM_embedding",
    "LinerRegressionApp Example": "dag_emb/LiR_embedding",
    "LogisticRegressionApp Example": "dag_emb/LoR_embedding",
    "Spark LabelPropagation Application": "dag_emb/LP_embedding",
    "MFApp Example": "dag_emb/MF_embedding",
    "Spark PCA Example": "dag_emb/PCA_embedding",
    "Spark PregelOperation Application": "dag_emb/PO_embedding",
    "Spark PageRank Application": "dag_emb/PR_embedding",
    "Spark StronglyConnectedComponent Application": "dag_emb/SCC_embedding",
    "Spark ShortestPath Application": "dag_emb/SP_embedding",
    "Spark SVDPlusPlus Application": "dag_emb/SVD_embedding",
    "SVM Classifier Example": "dag_emb/SVM_embedding",
    "Spark TriangleCount Application": "dag_emb/TC_embedding",
    "TeraSort": "dag_emb/TS_embedding"
}

for key in workload2file.keys():
    workload2file[key] = np.load(workload2file[key]).tolist()
    workload2emb[key] = np.loadtxt(workload2emb[key]).tolist()


def load_tfidf():
    workload2tfidf = lambda w: workload2file[w]
    return workload2tfidf


def load_emb():
    w2emb = lambda w: workload2file[w]
    return w2emb


def clean_code(s):
    if s is None or str(s).strip() is '':
        return []
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]+'
    line = re.sub(r, ' ', str(s))
    words = line.split(' ')
    words = list(filter(None, words))
    for word in words:
        word = word.strip()
        try:
            float(word)
            continue
        except:
            pass
    return words


def load_inst_code(filtered=False):
    if filtered:
        inst_code_dir = 'all_code_by_stage/all_code_filtered/'
        tfidf_vectorizer = joblib.load('all_code_by_stage/tfidf_all_filtered.pickle')
    else:
        inst_code_dir = 'all_code_by_stage/all_code/'
        tfidf_vectorizer = joblib.load('all_code_by_stage/tfidf_all.pickle')
    inst_code_dict = {}
    for app_name, short_name in workload_dict.items():
        app_dir = inst_code_dir + short_name
        stages = os.listdir(app_dir)
        all_vec = []
        for stage in stages:
            file = open(app_dir+'/'+stage, 'r', encoding='utf-8')
            code = file.read().replace("<SEP>", '')
            words = clean_code(code)
            tfidf_vec = tfidf_vectorizer.transform([' '.join(words)]).toarray()
            all_vec.append(tfidf_vec)
        all_vec = np.array(all_vec)
        avg_vec = np.average(all_vec, axis=0)
        inst_code_dict[app_name] = avg_vec
    return lambda w: inst_code_dict[w]


def get_workload_feat(data, row_dict):
    row_dict['AppId'] = data['AppId']
    row_dict['AppName'] = data['AppName']
    row_dict['Duration'] = data['Duration']
    for spark_param in data['SparkParameters']:
        if spark_param.find('spark.executor.memory') >= 0:
            spark_param = spark_param.strip().replace("g", "").replace("512m", "0.5")
        k, v = spark_param.split("=")
        row_dict[str(k)] = float(v.replace('g', '').replace('m', "").replace('k', '').replace("true", "1").replace("false", "0"))
    row_dict['rows'],  row_dict["cols"], row_dict["itr"], row_dict["partitions"] = 0, 0, 0, 0
    for workload_param in data['WorkloadConf']:
        workload_param = workload_param.split('#')[0]
        try:
            k, v = workload_param.split("=")
            if k == "numV" or k == "NUM_OF_EXAMPLES" or k == 'NUM_OF_SAMPLES' \
                    or k == 'NUM_OF_POINTS' or k == 'm' or k == 'NUM_OF_RECORDS':
                row_dict["rows"] = float(v)
            if k == "NUM_OF_FEATURES" or k == 'n':
                row_dict["cols"] = float(v)
            if k == "MAX_ITERATION":
                row_dict["itr"] = float(v)
            if k == "NUM_OF_PARTITIONS":
                row_dict["partitions"] = float(v)
        except:
            continue
    return row_dict


def build(data_dir, env_dic, writer):
    paths = os.listdir(data_dir)
    for path in paths:
        dataset_file = open(data_dir + path, encoding='utf-8')
        i = 0
        for line in dataset_file:
            strs = path.split('_')
            i += 1
            if i % 100 == 0:
                print(i)
            row_dict = {}
            data = json.loads(line)
            try:
                row_dict = get_workload_feat(data, row_dict)
            except:
                print('error')
                continue
            row_dict.update(env_dic)
            writer.writerow(row_dict)


def build_all(data_dir, csv_path):
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=all_fields)
    writer.writeheader()
    all_env = os.listdir(data_dir)
    for env_name in all_env:
        env_dic = envs[env_name]
        build(data_dir+env_name+'/', env_dic, writer)
    csv_file.close()


def df2npy(df, use_tf_idf=True, use_embedding=False, use_inst_code=False):
    Y = np.array(df['Duration'])
    names = df['AppName']
    df = df.drop(['AppId', 'AppName', 'Duration'], axis=1)
    X = df.values[:, 1:]
    if use_tf_idf:
        # AppName to tfidf
        workload2tfidf = load_tfidf()
        tfidf = names.apply(workload2tfidf).values.tolist()
        workload_vec = np.array(tfidf)
        workload_vec = workload_vec.squeeze(axis=1)
        X = np.concatenate((X, workload_vec), axis=1)
    if use_embedding:
        w2emb = load_emb()
        emb = names.apply(w2emb).values.tolist()
        emb_vec = np.array(emb)
        emb_vec = emb_vec.squeeze(axis=1)
        X = np.concatenate((X, emb_vec), axis=1)
    if use_inst_code:
        w2inst = load_inst_code(filtered=False)
        tfidf = names.apply(w2inst).values.tolist()
        tfidf_vec = np.array(tfidf)
        tfidf_vec = tfidf_vec.squeeze(axis=1)
        X = np.concatenate((X, tfidf_vec), axis=1)
    X, Y = shuffle(X, Y)
    return X, Y


def read_merged_data(merged_file_path, use_tf_idf=True):
    csv_data = pd.read_csv(merged_file_path, sep=',', low_memory=False)
    df = pd.DataFrame(csv_data)
    return df2npy(df, use_tf_idf=use_tf_idf)


def read_merged_data_with_emb(merged_file_path):
    csv_data = pd.read_csv(merged_file_path, sep=',', low_memory=False)
    df = pd.DataFrame(csv_data)
    return df2npy(df, use_tf_idf=False, use_embedding=True)


def read_merged_data_with_inst_code(merged_file_path):
    csv_data = pd.read_csv(merged_file_path, sep=',', low_memory=False)
    df = pd.DataFrame(csv_data)
    return df2npy(df, use_tf_idf=False, use_embedding=False, use_inst_code=True)


train_workloads = ['CC', 'DT', 'KM', 'LiR', 'LoR', 'LP', 'MF', 'PCA']
if __name__ == '__main__':
    # 从若干json数据中合并成一个csv数据集
    # build_all(data_dir='file/', csv_path='merged_file/dataset_train.csv')
    # build_all(data_dir='file_test_new/', csv_path='merged_file/dataset_test.csv')

    # 读取csv数据集
    # X, Y = read_merged_data('merged_file/dataset_train.csv', use_tf_idf=True)

    # load_inst_code()
    # X, Y = read_merged_data_with_inst_code('merged_file/dataset_train.csv')

    d = pd.read_csv('merged_file/dataset_train_8.csv')
    print(d["AppName"].value_counts())
    print()

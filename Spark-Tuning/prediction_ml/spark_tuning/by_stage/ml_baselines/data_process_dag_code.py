#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: data_process_dag_code.py 
@create: 2021/1/21 18:17 
"""
import re
import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib


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


def load_dag_code_by_stage():
    workload_stage2code = {}
    workloads = os.listdir(dag_code_dir)
    for workload in workloads:
        file = open(dag_code_dir+workload)
        stage2code = json.loads(file.read())
        workload_stage2code[workload] = stage2code
    return workload_stage2code


def clean_code(s):
    if s is None or str(s).strip() is '':
        return []
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]+'
    line = re.sub(r, ' ', str(s))
    words = line.split(' ')
    words = list(filter(None, words))
    for word in words:
        word = word.strip()
        # 过滤数字
        try:
            float(word)
            continue
        except:
            pass
    return words


def code2tfidf(code_list, corpus, n_feature):
    # tfidf_vectorizer = TfidfVectorizer(max_features=n_feature)
    # tfidf_vectorizer.fit(corpus)
    # joblib.dump(tfidf_vectorizer, 'saved_model/tfidf_dag.pickle')
    tfidf_vectorizer = joblib.load('saved_model/tfidf_dag.pickle')
    vec_list = []
    for code in tqdm(code_list):
        words = clean_code(code)
        tfidf_vec = tfidf_vectorizer.transform([' '.join(words)]).toarray()
        vec_list.append(tfidf_vec)
    vec_arr = np.array(vec_list)
    return vec_arr


def code2ngram(code_list, corpus, n_feature):
    cnt_vectorizer = CountVectorizer(max_features=n_feature, ngram_range=(2, 2))
    cnt_vectorizer.fit(corpus)
    vec_list = []
    for code in tqdm(code_list):
        words = clean_code(code)
        cnt_vec = cnt_vectorizer.transform([' '.join(words)]).toarray()
        vec_list.append(cnt_vec)
    vec_arr = np.array(vec_list)
    return vec_arr


def get_code_vec(workload, stage_id, vec_type, n_feature):
    workload_stage2code = load_dag_code_by_stage()
    corpus = []
    for w, stage in workload_stage2code.items():
        for s, sent in stage.items():
            corpus.append(sent)
    code_list = []
    for w, s in zip(workload, stage_id):
        try:
            code_txt = workload_stage2code[w][str(s)]
        except:
            code_txt = ''
        code_list.append(code_txt)
    if vec_type == 'tfidf':
        return code2tfidf(code_list, corpus, n_feature)
    elif vec_type == 'ngram':
        return code2ngram(code_list, corpus, n_feature)


def read_stage_dataset(dataset_path, vec_type, n_feature=128):
    csv_data = pd.read_csv(dataset_path, sep=',', low_memory=False)
    df = pd.DataFrame(csv_data)
    # workload + stage id -> TF-IDF / n-gram
    workload, stage_id = df['AppName'].apply(lambda x: workload_dict[x]), df['stage_id']
    tf_idf_vec = get_code_vec(workload, stage_id, vec_type, n_feature)
    code_vec = np.array(tf_idf_vec)
    Y = df['duration']
    X_df = df.drop(['AppId', 'AppName', 'Duration', 'code', 'duration'], axis=1)
    # X_df = X_df.drop(['input', 'output', 'read', 'write'], axis=1)   # drop metrics
    X = X_df.values[:, :]
    code_vec = code_vec.squeeze(axis=1)
    X = np.concatenate((X, code_vec), axis=1)
    # Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, Y


dag_code_dir = '../instrumentation/dag_code_by_stage/dag_code/'
if __name__ == '__main__':
    code_type = "tfidf"  # tfidf  ngram
    X, Y = read_stage_dataset(dataset_path='csv_dataset/dataset_by_stage.csv',
                              vec_type=code_type,  # tfidf  ngram
                              n_feature=128)
    np.save('npy_dataset_dag_code/X_' + code_type + '.npy', X)
    np.save('npy_dataset_dag_code/Y.npy', Y)


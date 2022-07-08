#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: data_process_all.py 
@create: 2020/12/29 9:48 
"""

import re
import os
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

workload2emb = {}
for key in workload_dict.keys():
    workload2emb[key] = np.loadtxt('dag_emb/' + workload_dict[key] + '_embedding').tolist()


def load_emb():
    w2emb = lambda w: workload2emb[w]
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


def code2tfidf(code_list, corpus, n_feature):
    # corpus = []
    # for code in tqdm(code_list):
    #     words = clean_code(code)
    #     corpus.append(' '.join(words))
    # tfidf_vectorizer = TfidfVectorizer(max_features=n_feature)
    # tfidf_vectorizer.fit(corpus)
    # joblib.dump(tfidf_vectorizer, 'saved_model/tfidf_all.pickle')

    if is_filtered:
        tfidf_vectorizer = joblib.load('saved_model/tfidf_all_filtered.pickle')
    else:
        tfidf_vectorizer = joblib.load('saved_model/tfidf_all.pickle')
    # print(tfidf_vectorizer.vocabulary_)
    vec_list = []
    for code in code_list:
        words = clean_code(code)
        tfidf_vec = tfidf_vectorizer.transform([' '.join(words)]).toarray()
        vec_list.append(tfidf_vec)
    tfidf_labels = tfidf_vectorizer.get_feature_names()
    vec_arr = np.array(vec_list)
    return vec_arr, tfidf_labels


def code2ngram(code_list, corpus, n_feature):
    cnt_vectorizer = CountVectorizer(max_features=n_feature, ngram_range=(2, 2))
    cnt_vectorizer.fit(corpus)
    vec_list = []
    for code in code_list:
        words = clean_code(code)
        cnt_vec = cnt_vectorizer.transform([' '.join(words)]).toarray()
        vec_list.append(cnt_vec)
    vec_arr = np.array(vec_list)
    return vec_arr


def load_all_code_by_stage():
    workload_stage2code = {}
    workloads = os.listdir(all_code_dir)
    for workload in workloads:
        stage2code = {}
        stages = os.listdir(all_code_dir+workload)
        # print(workload)
        tokens_len = []
        for stage_id in stages:
            file = open(all_code_dir+workload+'/'+stage_id)
            stage_code = file.read().replace('<SEP>', '')
            tokens = clean_code(stage_code)
            tokens_len.append(len(tokens))
            stage2code[stage_id] = stage_code
        workload_stage2code[workload] = stage2code
        # print(np.mean(tokens_len))
    return workload_stage2code


def get_code_vec(workload, stage_id, vec_type, n_feature):
    workload_stage2code = load_all_code_by_stage()
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


def read_stage_dataset(dataset_path, vec_type, n_feature=128, use_emb=False):
    csv_data = pd.read_csv(dataset_path, sep=',', low_memory=False)
    df = pd.DataFrame(csv_data)
    names = df['AppName']
    # workload + stage id -> TF-IDF / n-gram
    workload, stage_id = df['AppName'].apply(lambda x: workload_dict[x]), df['stage_id']
    tf_idf_vec, tf_idf_labels = get_code_vec(workload, stage_id, vec_type, n_feature)
    code_vec = np.array(tf_idf_vec)
    Y = df['duration']
    X_df = df.drop(['AppId', 'AppName', 'Duration', 'code', 'duration'], axis=1)
    # X_df = X_df.drop(['input', 'output', 'read', 'write'], axis=1)   # drop metrics
    X_labels = X_df.columns
    X = X_df.values[:, :]
    code_vec = code_vec.squeeze(axis=1)
    X = np.concatenate((X, code_vec), axis=1)
    X_labels = np.concatenate((X_labels, tf_idf_labels))
    # Scaling
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    if use_emb:
        w2emb = load_emb()
        emb = names.apply(w2emb).values.tolist()
        emb_vec = np.array(emb)
        # emb_vec = emb_vec.squeeze(axis=1)
        X = np.concatenate((X, emb_vec), axis=1)
    return X, Y, X_labels


all_code_dir = '../instrumentation/all_code_by_stage/all_code/'
# all_code_dir = '../instrumentation/all_code_by_stage/all_code_filtered/'
code_type = "tfidf"   # tfidf ngram
is_filtered = False
if is_filtered:
    all_code_dir = all_code_dir.replace("all_code/", "all_code_filtered/")

if __name__ == '__main__':

    X, Y, X_labels = read_stage_dataset(dataset_path='csv_dataset/dataset_by_stage.csv',
                                        vec_type=code_type,  # tfidf  ngram
                                        n_feature=128, use_emb=True)
    if is_filtered:
        np.save('npy_dataset_all_filtered/X_' + code_type + '_emb_13.npy', X)
        np.save('npy_dataset_all_filtered/Y_emb_13.npy', Y)
    else:
        np.save('npy_dataset_all/X_' + code_type + '_emb_13.npy', X)
        np.save('npy_dataset_all/Y_emb_13.npy', Y)




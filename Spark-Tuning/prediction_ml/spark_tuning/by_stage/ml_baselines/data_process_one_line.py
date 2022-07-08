#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: data_process.py 
@create: 2020/11/30 16:06 
"""

import re
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


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


def code2tfidf(code_list):
    corpus = []
    for code in code_list:
        words = clean_code(code)
        corpus.append(' '.join(words))
    tfidf_vectorizer = TfidfVectorizer(max_features=64)
    tfidf_vectorizer.fit(corpus)
    # print(tfidf_vectorizer.vocabulary_)
    vec_list = []
    for code in tqdm(code_list):
        words = clean_code(code)
        tfidf_vec = tfidf_vectorizer.transform([' '.join(words)]).toarray()
        vec_list.append(tfidf_vec)
    vec_arr = np.array(vec_list)
    return vec_arr


def code2ngram(code_list):
    corpus = []
    for code in code_list:
        words = clean_code(code)
        corpus.append(' '.join(words))
    cnt_vectorizer = CountVectorizer(max_features=64, ngram_range=(2, 2))
    cnt_vectorizer.fit(corpus)
    # print(cnt_vectorizer.vocabulary_)
    vec_list = []
    for code in tqdm(code_list):
        words = clean_code(code)
        cnt_vec = cnt_vectorizer.transform([' '.join(words)]).toarray()
        vec_list.append(cnt_vec)
    vec_arr = np.array(vec_list)
    return vec_arr


def read_stage_dataset(dataset_path, code_type):
    csv_data = pd.read_csv(dataset_path, sep=',', low_memory=False)
    df = pd.DataFrame(csv_data)
    # ****code -> TF-IDF
    if code_type == "tfidf":
        code_list = df['code'].tolist()
        vec_arr = code2tfidf(code_list)
    # ****code -> n-gram
    elif code_type == "ngram":
        code_list = df['code'].tolist()
        vec_arr = code2ngram(code_list)
    else:
        vec_arr = None
    Y = df['duration']
    X_df = df.drop(['AppId', 'AppName', 'Duration', 'code', 'duration'], axis=1)
    # X_df = X_df.drop(['input', 'output', 'read', 'write'], axis=1)   # drop metrics
    X = X_df.values[:, 0:]
    if vec_arr is not None:
        vec_arr = vec_arr.squeeze(axis=1)   # code tf-idf
        X = np.concatenate((X, vec_arr), axis=1)
    return X, Y


if __name__ == '__main__':
    code_type = "none"  # tfidf  ngram none
    X, Y = read_stage_dataset('csv_dataset/dataset_by_stage.csv', code_type)
    # np.save('npy_dataset_one_line/X_' + code_type + '.npy', X)
    # np.save('npy_dataset_one_line/Y.npy', Y)
    np.save('npy_dataset_no_code/X_13.npy', X)
    np.save('npy_dataset_no_code/Y_13.npy', Y)

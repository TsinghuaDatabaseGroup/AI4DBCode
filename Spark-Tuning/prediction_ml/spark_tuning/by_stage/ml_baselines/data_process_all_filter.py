#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: data_process_all_filter.py 
@create: 2020/12/29 15:40 
"""

import os
import numpy as np
import heapq
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest


def build_vocab():
    all_words = []
    old_log_dir = '../instrumentation/log/'
    workloads = os.listdir(old_log_dir)
    for workload in workloads:
        stages = os.listdir(old_log_dir + workload)
        for stage_path in stages:
            file = open(old_log_dir + workload + '/' + stage_path)
            word_freq = file.read().split('\n')
            words = [x.split('=')[0] for x in word_freq]
            all_words.extend(words)
            file.close()
    word_count = {}
    for word in all_words:
        if word not in word_count:
            word_count[word] = 0
        word_count[word] += 1
    word_count = list(word_count.items())
    word_count.sort(key=lambda k: k[1], reverse=True)
    w2i, i2w = {}, {}
    i = 1
    for word_pair in word_count:
        w2i[word_pair[0]] = i
        i2w[str(i)] = word_pair[0]
        i += 1
    cnt_vectorizer = CountVectorizer()
    cnt_vectorizer.fit(all_words)
    return w2i, i2w, cnt_vectorizer


def read_log_for_mic(w2i):
    all_sample_for_mic = []
    all_y_for_mic = []
    workload_stage_vec = {}
    workloads = os.listdir(old_log_dir)
    for w_idx, workload in enumerate(workloads):
        stages = os.listdir(old_log_dir + workload)
        stage2vec = {}
        for stage_path in stages:
            file = open(old_log_dir + workload + '/' + stage_path)
            word_freq = file.read().split('\n')
            word_freq = list(filter(None, word_freq))
            words = [x.split('=')[0] for x in word_freq]
            freqs = [x.split('=')[1] for x in word_freq]
            stage_id = stage_path.split('.')[0].split('_')[1]
            vec = [0 for _ in range(len(w2i) + 1)]
            for i in range(len(words)):
                vec[w2i[words[i]]] = freqs[i]
            stage2vec[stage_id] = vec
            all_sample_for_mic.append(vec)
            all_y_for_mic.append(w_idx)
            file.close()
        workload_stage_vec[workload] = stage2vec
    return all_sample_for_mic, all_y_for_mic


def filter_feature():
    w2i, i2w, cnt_vectorizer = build_vocab()
    all_sample_for_mic, all_y_for_mic = read_log_for_mic(w2i)
    # mic分类检验
    result = MIC(all_sample_for_mic, all_y_for_mic).tolist()
    best_feat_idx = set(map(result.index, heapq.nlargest(32, result)))
    # 得到最好特征id后，过滤
    workloads = os.listdir(old_log_dir)
    for w_idx, workload in enumerate(workloads):
        stages = os.listdir(old_log_dir + workload)
        stage2vec = {}
        for stage_path in stages:
            file = open(old_log_dir + workload + '/' + stage_path)
            new_file = open(new_log_dir + workload + '/' + stage_path, 'w')
            word_freq = file.read().split('\n')
            word_freq = list(filter(None, word_freq))
            for line in word_freq:
                w, f = line.split('=')
                if w2i[w] in best_feat_idx:
                    new_file.write(line+'\n')
            file.close()
            new_file.close()
    print()


old_log_dir = '../instrumentation/log/'
new_log_dir = '../instrumentation/log_filtered/'
if __name__ == '__main__':
    filter_feature()
    print()


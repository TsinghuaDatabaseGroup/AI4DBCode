#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: code_process.py 
@create: 2020/11/14 15:52 
"""

import os
import re
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def code2tfidf(text_dir):
    def clean_code(s):
        line = re.sub(r, ' ', contents)
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
    corpus = []
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]+'
    paths = os.listdir(text_dir)
    for path in paths:
        contents = Path(text_dir + path).read_text().replace(";", '')
        words = clean_code(contents)
        corpus.append(' '.join(words))
    tfidf_vectorizer = TfidfVectorizer(max_features=32)
    tfidf_vectorizer.fit(corpus)
    print(tfidf_vectorizer.vocabulary_)
    for path in paths:
        contents = Path(text_dir + path).read_text().replace(";", '')
        words = clean_code(contents)
        tfidf_vec = tfidf_vectorizer.transform([' '.join(words)]).toarray()
        # tfidf_file = open('tfidf/' + path.split('/')[1], 'w', encoding='utf-8')
        np.save('tf-idf/' + path.split('.')[0] + '.npy', tfidf_vec)


if __name__ == '__main__':

    code2tfidf('bench_code/')

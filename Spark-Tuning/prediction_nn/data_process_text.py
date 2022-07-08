#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: data_process_text.py 
@create: 2020/12/21 15:32 
"""

import os
import re
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="get code dict")

    parser.add_argument('all_code_dir', type=str, help='dir of all code of stage')

    return parser.parse_args()

args = parse_args()

def load_all_code_by_stage():
    workload_stage2code = {}
    workloads = os.listdir(all_code_dir)
    for workload in workloads:
        stage2code = {}
        stages = os.listdir(all_code_dir+workload)
        for stage_id in stages:
            file = open(all_code_dir+workload+'/'+stage_id)
            stage_code = file.read().replace('<SEP>', '')
            stage2code[stage_id] = stage_code
        workload_stage2code[workload] = stage2code
    return workload_stage2code


def build_vocab_code():
    workload_stage2code = load_all_code_by_stage()
    word_count = {}
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]+'
    vocab_file = open('all_code_seq_ids/code.vocab', 'w', encoding='utf-8')
    for workload, stage2code in workload_stage2code.items():
        for stage, code in stage2code.items():
            code = re.sub(r, ' ', code)
            words = code.split(' ')
            words = list(filter(None, words))
            for word in words:
                word = word.strip()
                try:
                    float(word)
                    continue
                except:
                    pass
                if word is not '' and word is not '\n' and word is not '\t':
                    if word not in word_count:
                        word_count[word] = 0
                    word_count[word] += 1
    word_count = list(word_count.items())
    word_count.sort(key=lambda k: k[1], reverse=True)
    for word_pair in word_count:
        vocab_file.write(word_pair[0] + '\t' + str(word_pair[1]) + '\n')
    vocab_file.close()
    print()


def get_w2i(vocab_path='all_code_seq_ids/code.vocab'):
    vocab_file = open(vocab_path, encoding='utf-8')
    w2i = {BLANK_WORD: 0, UNKNOWN_WORD: 1, BOS_WORD: 2, EOS_WORD: 3}
    i = 4
    for pair in vocab_file:
        v = pair.split('\t')[0]
        w2i[v] = i
        i += 1
    return w2i


def fix_length(seq,):
    seq = [BOS_WORD] + seq
    if len(seq) >= SEQ_LEN:
        seq = seq[0:SEQ_LEN-1] + [EOS_WORD]
    else:
        seq = np.concatenate((seq, [EOS_WORD], [BLANK_WORD for _ in range(SEQ_LEN - 1 - len(seq))]))
    return seq


def code2ids(contents):
    w2i = get_w2i()
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]+'
    try:
        line = re.sub(r, ' ', contents)
        words = line.split(' ')
        words = list(filter(None, words))
        words = fix_length(words)
        ids = list(map(lambda x: w2i[x] if x in w2i.keys() else 1, words))
    except:
        ids = [0 for _ in range(SEQ_LEN)]
    return ids


def load_all_code_ids():
    workload_stage2code = load_all_code_by_stage()
    for workload, stage2code in workload_stage2code.items():
        for stage, code in stage2code.items():
            stage2code[stage] = code2ids(stage2code[stage])
    return workload_stage2code


BLANK_WORD = '<PAD>'
UNKNOWN_WORD = '<UNK>'
BOS_WORD = '<S>'
EOS_WORD = '</S>'
SEQ_LEN = 1000
# all_code_dir = 'all_code/'
all_code_dir = args.all_code_dir
if __name__ == '__main__':
    build_vocab_code()
    # ids = code2ids('val a = 1')
    w = load_all_code_ids()
    print('end')


#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: config.py 
@create: 2021/5/21 21:03 
"""


class Config(object):

    hidden_dim = 128

    # for code lstm
    code_seq_len = 50
    code_vocab_size = 2412
    code_emb_dim = 128
    code_hidden_dim = 128
    code_lstm_layers = 2

    # for model train
    lr = 0.0005  # 8机 0.0005
    lr_decay_epochs = 20
    lr_decay_ratio = 0.8
    epoch_num = 200
    batch_size = 128

    # transfer learning
    trans_batch_size = 1024



#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: light_gbm_rank.py 
@create: 2021/5/7 18:28 
"""

import numpy as np
import lightgbm as lgb

# 生成group list  官方文档有介绍
# 比如[10 20 30]表示data里面前10个样本属于一个group，其后20个属于一个group，最后30个属于一个group

n_size = 20  # 每一个query对应的doc数量 我的每一个group具有相同的doc数量
dgroup_train = np.array([n_size for _ in range(5)]).flatten()  # [20 20 20 20 20]
dgroup_valid = np.array([n_size for _ in range(5)]).flatten()  # [20 20 20 20 20]
lgb_train = lgb.Dataset(dtrain, dtrain_y, group=dgroup_train, free_raw_data=False)
lgb_valid = lgb.Dataset(dvalid, dvalid_y, group=dgroup_valid, free_raw_data=False)

lgb_train.set_group(dgroup_train)
lgb_valid.set_group(dgroup_valid)
params = {
    'objective': 'lambdarank',
    'boosting_type': 'gbdt',
    'num_trees': 30,
    'num_leaves': 128,
    'feature_fraction': 0.2,
    'bagging_fraction': 0.2,
    'max_bin': 256,
    'learning_rate': 0.1,
    'metric': 'ndcg'
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1500,
                valid_sets=[lgb_train, lgb_valid],
                early_stopping_rounds=150
                )

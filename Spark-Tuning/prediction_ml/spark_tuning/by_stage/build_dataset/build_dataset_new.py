#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: build_dataset_new.py 
@create: 2021/3/12 15:30 
"""
import os
import json
import csv
import pandas as pd
from copy import deepcopy

all_fields = ['AppId', 'AppName', 'Duration',
              'spark.default.parallelism', 'spark.driver.cores', 'spark.driver.memory', 'spark.driver.maxResultSize', 'spark.executor.instances', 'spark.executor.cores', 'spark.executor.memory', 'spark.executor.memoryOverhead', 'spark.files.maxPartitionBytes', 'spark.memory.fraction', 'spark.memory.storageFraction', 'spark.reducer.maxSizeInFlight', 'spark.shuffle.compress', 'spark.shuffle.file.buffer', 'spark.shuffle.spill.compress',
              'rows', 'cols', 'itr', 'partitions',
              'stage_id', 'duration', 'input', 'output', 'read', 'write', 'code',
              'node_num', 'cpu_cores', 'cpu_freq', 'mem_size', 'mem_speed', 'net_width']

env_1_1000M = {"node_num": 1, "cpu_cores": 16, "cpu_freq": 3.2, "mem_size": 64, "mem_speed": 2400, "net_width": 1000}
env_3_1000M = {"node_num": 3, "cpu_cores": 16, "cpu_freq": 3.2, "mem_size": 64, "mem_speed": 2400, "net_width": 1000}
env_8_1000M = {"node_num": 8, "cpu_cores": 16, "cpu_freq": 2.9, "mem_size": 16, "mem_speed": 2666, "net_width": 1000}
env_8_10000M = {"node_num": 8, "cpu_cores": 16, "cpu_freq": 2.9, "mem_size": 16, "mem_speed": 2666, "net_width": 10000}

envs = {
    "1_1000M": env_1_1000M, "3_1000M": env_3_1000M, "8_1000M": env_8_1000M, "8_10000M": env_8_10000M
}


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


def load_code(code_file):
    txt = code_file.read()
    json_data = json.loads(txt)
    return json_data


def process_one_workload(data_dir, path, merged_df):
    dataset_file = open(data_dir + path, encoding='utf-8')
    code_file = open('../code_by_stage/' + path.split('_')[0] + '.json', encoding='utf-8')
    code_dict = load_code(code_file)
    i = 0
    for line in dataset_file:
        i += 1
        if i % 100 == 0:
            print(i)
        row_dict = {}
        data = json.loads(line)
        try:
            row_dict = get_workload_feat(data, row_dict)
        except:
            print('error')
        for stage_id, info in data['StageInfo'].items():
            stage_dict = deepcopy(row_dict)
            stage_dict['stage_id'] = stage_id
            stage_dict.update(info)
            stage_dict['code'] = code_dict[stage_id].strip() if stage_id in code_dict.keys() else ''
            merged_df = merged_df.append(stage_dict, ignore_index=True)
    return merged_df


def build(data_dir, env_dic, writer):
    paths = os.listdir(data_dir)
    workload_count = {}
    stage_count = {}
    for path in paths:
        dataset_file = open(data_dir + path, encoding='utf-8')
        strs = path.split('_')
        if len(strs) == 6 or len(strs) == 7:
            short_name = strs[4]
            code_file = open('one_line_code_by_stage/' + short_name + '.json', encoding='utf-8')
        else:
            short_name = strs[1]
            code_file = open('one_line_code_by_stage/' + short_name, encoding='utf-8')
        code_dict = load_code(code_file)
        i = 0
        j = 0
        for line in dataset_file:
            i += 1
            row_dict = {}
            data = json.loads(line)
            try:
                row_dict = get_workload_feat(data, row_dict)
            except:
                print('error')
            if short_name == "SVD":
                print(len(data['StageInfo'].items()))
            for stage_id, info in data['StageInfo'].items():
                j += 1
                stage_dict = deepcopy(row_dict)
                stage_dict['stage_id'] = stage_id
                stage_dict.update(info)
                stage_dict['code'] = code_dict[stage_id].strip() if stage_id in code_dict.keys() else ''
                stage_dict.update(env_dic)
                writer.writerow(stage_dict)
        if short_name in workload_count:
            workload_count[short_name] = workload_count[short_name] + i
        else:
            workload_count[short_name] = i
        if short_name in stage_count:
            stage_count[short_name] = stage_count[short_name] + j
        else:
            stage_count[short_name] = j
    print(stage_count)
    print(workload_count)


def build_all(data_dir, csv_path):
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=all_fields)
    writer.writeheader()
    all_env = os.listdir(data_dir)
    for env_name in all_env:
        env_dic = envs[env_name]
        build(data_dir+env_name+'/', env_dic, writer)
    csv_file.close()


if __name__ == '__main__':
    build_all(data_dir='dataset_by_workload/', csv_path='dataset_by_stage_merged/dataset_by_stage_8.csv')
    # build_all(data_dir='dataset_test_new/', csv_path='dataset_by_stage_merged/dataset_test_1.csv')

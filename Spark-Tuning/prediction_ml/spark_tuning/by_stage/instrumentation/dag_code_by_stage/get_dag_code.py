#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: get_dag_code.py 
@create: 2021/1/21 16:29 
"""

import os
import json


def search_in_lib(file_name, line, root='../../../spark-lib/spark-2.4.6/'):
    paths = os.listdir(root)
    for item in paths:
        path = os.path.join(root, item)
        if os.path.isdir(path):
            res = search_in_lib(file_name, line, path)
            if res != '':
                return res
        elif item == file_name:
            tgt_file = open(path, encoding='utf-8')
            try:
                code = tgt_file.readlines()[int(line)-1]
            except:
                return ''
            # print('code: ', code)
            tgt_file.close()
            return code
        else:
            continue
    return ''


def search_in_bench(file_name, line):
    paths = os.listdir('bench-code/')
    for item in paths:
        if item == file_name:
            tgt_file = open('bench-code/' + item, encoding='utf-8')
            code = tgt_file.readlines()[int(line) - 1]
            # print('code: ', code)
            tgt_file.close()
            return code
    return ''


def get_dag_code(log_path):
    workload_name = log_path.split('/')[-1]
    log_file = open(log_path)
    data = json.loads(log_file.read())
    stage_dag_code_dict = {}
    for key, dag in data['dags'].items():
        stage_id = key.split('_')[-1]
        code_all = ''
        for node in dag:
            call_site = node['call_site']
            file_line = call_site.split(' ')[-1]
            file_name, line = file_line.split(':')
            code = search_in_lib(file_name, line)
            if code == '':
                code = search_in_bench(file_name, line)
            code_all += ' ' + code
        stage_dag_code_dict[stage_id] = code_all
    new_file = open('dag_code/' + workload_name, 'w', encoding='utf-8')
    new_file.write(json.dumps(stage_dag_code_dict))
    new_file.close()
    print()


if __name__ == '__main__':
    workload_names = os.listdir("raw_logs")
    for name in workload_names:
        get_dag_code('raw_logs/' + name)


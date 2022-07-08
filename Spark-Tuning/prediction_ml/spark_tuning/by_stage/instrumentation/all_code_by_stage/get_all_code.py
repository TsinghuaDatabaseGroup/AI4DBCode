#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: get_all_code.py 
@create: 2020/12/26 16:21 
"""

import os

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="get stage code")

    parser.add_argument('base_log_dir', type=str, help='files from instrumentation')
    parser.add_argument('all_code_dir', type=str, help='dir of all code of stage')
    return parser.parse_args()

args = parse_args()

def search_in_lib(guide, file_name, start_line, end_line, root='../../../spark-lib/spark-2.4.6/'):
    paths = os.listdir(root)
    for item in paths:
        path = os.path.join(root, item)
        if os.path.isdir(path):
            res = search_in_lib(guide, file_name, start_line, end_line, path)
            if res != '':
                return res
        elif item == file_name and path.find(guide) >= 0:
            tgt_file = open(path, encoding='utf-8')
            file_code = tgt_file.readlines()
            code = file_code[int(start_line)-1:int(end_line)]
            # print('code: ', code)
            tgt_file.close()
            return code
        else:
            continue
    return ''


def get_code(class_name, file_name, start_line, end_line):
    class_file_name = class_name.split('$')[0]
    idx = class_file_name.rfind('/')
    guide, _ = class_file_name[0:idx].replace('/', '\\'), class_file_name[idx+1:]+'.scala'
    code = search_in_lib(guide, file_name, start_line, end_line)
    return code


def get_code_a_stage(workload, stage):
    file = open(base_log_dir+workload+'/inst_'+str(stage)+'.log')
    write_file = open(all_code_dir+workload+'/'+str(stage), 'w')
    for line in file:
        try:
            key, _ = line.split('=')
            class_method, file_name, start_line, end_line = key.split('-')
            class_name, method_name = class_method.split('.')
            code = get_code(class_name, file_name, start_line, end_line)
            write_file.write(' '.join(code) + '<SEP>\n')
        except:
            continue
    write_file.close()
    file.close()


def get_code_a_workload(workload):
    log_paths = os.listdir(base_log_dir+workload)
    for path in log_paths:
        stage = path.split('.')[0].split('_')[1]
        get_code_a_stage(workload, stage)


# base_log_dir = '../log/'
# all_code_dir = 'all_code/'

# base_log_dir = '../log_filtered/'
# all_code_dir = 'all_code_filtered/'
base_log_dir = args.base_log_dir
all_code_dir = args.all_code_filtered
spark_lib_dir = '../../../spark-lib/spark-2.4.6/'
if __name__ == '__main__':
    workloads = os.listdir(base_log_dir)
    for workload in workloads:
        get_code_a_workload(workload)
    print()
    # get_code_a_workload('PCA')

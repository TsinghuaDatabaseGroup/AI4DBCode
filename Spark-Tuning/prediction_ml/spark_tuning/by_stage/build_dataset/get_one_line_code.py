#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: get_one_line_code.py 
@create: 2020/11/26 15:05 
"""

import os
import json


def search_in_lib(guide, file_name, line, root='../../spark-lib/spark-2.4.6/'):
    paths = os.listdir(root)
    for item in paths:
        path = os.path.join(root, item)
        if os.path.isdir(path):
            res = search_in_lib(guide, file_name, line, path)
            if res != '':
                return res
        elif item == file_name and path.find(guide) >= 0:
            tgt_file = open(path, encoding='utf-8')
            code = tgt_file.readlines()[int(line)-1]
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


def get_code(history_path, stage_code_path):
    stage_code_file = open(stage_code_path, 'w', encoding='utf-8')
    stage_code = {}
    if history_path.endswith('inprogress'):
        return
    his_file = open(history_path, encoding='utf-8')
    for line in his_file:
        try:
            line_json = json.loads(line)
        except:
            print('json error')
            continue
        if line_json['Event'] == 'SparkListenerStageSubmitted':
            stage_id = line_json['Stage Info']['Stage ID']
            code_detail = line_json['Stage Info']['Details']
            code_site = code_detail.split('\n')[1]
            guide_words = code_site.split('(')[0].split('.')[:-2]
            guide = '\\'.join(guide_words)
            code_site = code_site.split('(')[1][:-1]
            file_name, line = code_site.split(':')
            code = search_in_lib(guide, file_name, line)
            if code == '':
                search_in_bench(file_name, line)
            stage_code[stage_id] = code
    stage_code_file.write(json.dumps(stage_code))
    stage_code_file.close()


if __name__ == '__main__':
    workload_names = os.listdir("raw-log")
    for name in workload_names:
        get_code(history_path="raw-log/" + name, stage_code_path="one_line_code_by_stage/" + name + ".json")


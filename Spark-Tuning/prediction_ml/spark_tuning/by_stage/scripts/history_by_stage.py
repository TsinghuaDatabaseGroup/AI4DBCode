#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: zzz_jq
# Time: 2020/9/4 13:30

import os
import json


def read_dag(line_json):
    stage_id = line_json['Stage Info']['Stage ID']
    dag_nodes = line_json['Stage Info']['RDD Info']
    final_nodes = []
    for node in dag_nodes:
        processed_node = {'rdd_id': node['RDD ID'],
                          'name': node['Name'],
                          'scope': '' if 'Scope' not in node else json.loads(node['Scope']),
                          'parent_ids': node['Parent IDs']}
        final_nodes.append(processed_node)
    return stage_id, final_nodes


def gen_data(history_dir, dataset_path, env_path):
    env_list = []
    with open(env_path, 'r', encoding='utf-8') as env_file:
        for line in env_file.readlines():
            if '=' in line and not line.startswith('#'):
                env_list.append(line.replace('\n', ''))

    dataset_file = open(dataset_path, 'w', encoding='utf-8')
    os.chdir(history_dir)
    his_file_list = os.listdir(history_dir)

    his_file_list.sort()

    for path in his_file_list:
        if path.endswith('inprogress'):
            continue
        his_file = open(path, encoding='utf-8')
        one_data = {}
        dags = {}
        all_stage_info = {}
        cur_metrics, cur_stage = {'input':0, 'output':0, 'read':0, 'write':0}, 0
        start_timestamp = None
        end_timestamp = None
        for line in his_file:
            try:
                line_json = json.loads(line)
            except:
                print('json错误')
                continue
            if line_json['Event'] == 'SparkListenerTaskEnd':
                if line_json['Stage ID'] != cur_stage:
                    cur_metrics, cur_stage = {'input':0, 'output':0, 'read':0, 'write':0}, line_json['Stage ID']
                try:
                    cur_metrics['input'] += line_json['Task Metrics']['Input Metrics']['Bytes Read']
                    cur_metrics['output'] += line_json['Task Metrics']['Output Metrics']['Bytes Written']
                    cur_metrics['read'] += (line_json['Task Metrics']['Shuffle Read Metrics']['Remote Bytes Read'] +
                                            line_json['Task Metrics']['Shuffle Read Metrics']['Local Bytes Read'])
                    cur_metrics['write'] += line_json['Task Metrics']['Shuffle Write Metrics']['Shuffle Bytes Written']
                except:
                    print('metrics key error')
                    continue
            if line_json['Event'] == 'SparkListenerEnvironmentUpdate':
                java_command = line_json['System Properties']['sun.java.command']
                conf_list = java_command.split("--class")[0].split("--conf")[1:]
                other_conf_list = java_command.split("--class")[1].split("/")[0].split("--")[1:]
                for element in other_conf_list:
                    element = element.strip().replace(' ', '=')
                    conf_list.append(element)
                one_data['SparkParameters'] = conf_list
            if line_json['Event'] == 'SparkListenerApplicationStart':
                start_timestamp = line_json['Timestamp']
                one_data['AppName'] = line_json['App Name']
                one_data['AppId'] = line_json['App ID']
            if line_json['Event'] == 'SparkListenerApplicationEnd':
                end_timestamp = line_json['Timestamp']
            # if line_json['Event'] == 'SparkListenerStageSubmitted':
            #    stage_id, dag = read_dag(line_json)
            #    dags['dag_stage_' + str(stage_id)] = dag

            if line_json['Event'] == 'SparkListenerStageCompleted':
                # stage_id = 'dag_stage_' + str(line_json['Stage Info']['Stage ID'])
                try:
                    stage_id = line_json['Stage Info']['Stage ID']
                    all_stage_info[stage_id] = cur_metrics
                    stage_start = line_json['Stage Info']['Submission Time']
                    stage_end = line_json['Stage Info']['Completion Time']
                    all_stage_info[stage_id]['duration'] = int(stage_end) - int(stage_start)
                except:
                    print('stage duration key error')
                    continue
        if end_timestamp and start_timestamp:
            duration = int(end_timestamp) - int(start_timestamp)
            one_data['WorkloadConf'] = env_list
            one_data['Duration'] = duration
        else:
            print("出错，未能获取运行时长")
        # one_data['dags'] = dags
        one_data['StageInfo'] = all_stage_info
        dataset_file.write(json.dumps(one_data, sort_keys=True) + '\n')


if __name__ == '__main__':
    gen_data(history_dir="/home/jqzhuang/log-all/PCA_4_param/",
             dataset_path="/home/jqzhuang/dataset/PCA_by_stage.json",
             env_path="../conf/env.sh")

""" 
@author: jdfeng
@contact: jdefeng@stu.xmu.edu.cn
@software: PyCharm 
@file: main.py 
@create: 2020/10/9 17:52 
"""

import os
import json

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="Parse log")

    parser.add_argument('spark_bench_conf_path', type=str, help='path to sparkbench conf')
    parser.add_argument('history_dir', type=str, help='path to sparkbench log on local')
    parser.add_argument('dataset_path', type=str, help='path to result')
    return parser.parse_args()

args = parse_args()


spark_bench_path = args.spark_bench_path

spark_conf_names = ['spark.default.parallelism', 'spark.driver.cores', 'spark.driver.memory',
                    'spark.driver.maxResultSize',
                    'spark.executor.instances', 'spark.executor.cores', 'spark.executor.memory',
                    'spark.executor.memoryOverhead',
                    'spark.files.maxPartitionBytes', 'spark.memory.fraction', 'spark.memory.storageFraction',
                    'spark.reducer.maxSizeInFlight',
                    'spark.shuffle.compress', 'spark.shuffle.file.buffer', 'spark.shuffle.spill.compress']

workload = ['ConnectedComponent', 'DecisionTree', 'KMeans', 'LabelPropagation', 'LinearRegression',
            'LogisticRegression', 'PageRank',
            'PCA', 'PregelOperation', 'ShortestPaths', 'StronglyConnectedComponent', 'SVM', 'Terasort', 'TriangleCount']

stage_count = [19, 16, 14, 87, 9, 10, 23, 7, 57, 7, 683, 17, 4, 11]


def read_dag(line_json):
    stage_id = line_json['Stage Info']['Stage ID']
    dag_nodes = line_json['Stage Info']['RDD Info']
    final_nodes = []
    for node in dag_nodes:
        # print(node)
        processed_node = {'rdd_id': node['RDD ID'],
                          'name': node['Name'],
                          'scope': '' if 'Scope' not in node else json.loads(node['Scope']),
                          'parent_ids': node['Parent IDs']}
        final_nodes.append(processed_node)
    return stage_id, final_nodes


def get_env(i):
    env_list = []
    env_path = spark_bench_path + workload[i] + "/conf/env.sh"
    with open(env_path, 'r', encoding='utf-8') as env_file:
        for line in env_file.readlines():
            if '=' in line and not line.startswith('#'):
                env_list.append(line.replace('\n', ''))
    return env_list


def gen_data(history_dir, dataset_path):
    count = 0
    env_list = []
    env_list_all = {}
    workload_stage_count_list = {}
    workload_stage_count = {workload[i]: stage_count[i] for i in range(len(workload))}
    env_list_all = {workload[i]: get_env(i) for i in range(len(workload))}
    print(len(env_list_all))
    dataset_file = open(dataset_path, 'w', encoding='utf-8')
    os.chdir(history_dir)
    his_file_list = os.listdir(history_dir)
    # 对历史记录按文件名排序
    his_file_list.sort()
    print(len(his_file_list))
    # 逐条进行读取
    for path in his_file_list:
        print(path)
        if path.endswith('inprogress'):
            continue
        his_file = open(path, encoding='utf-8')
        # 处理一条历史记录
        one_data = {}
        dags = {}
        all_stage_info = {}
        cur_metrics, cur_stage = {'input': 0, 'output': 0, 'read': 0, 'write': 0}, 0
        start_timestamp = None
        end_timestamp = None
        for line in his_file:
            try:
                line_json = json.loads(line)
            except:
                print('json错误')
                continue
            # 统计每个stage的shuffle read/write、input/output，需要每个task累加得到stage
            if line_json['Event'] == 'SparkListenerTaskEnd':
                cur_stage = line_json['Stage ID']
                # 新的stage
                if line_json['Stage ID'] not in all_stage_info:
                    all_stage_info[cur_stage] = {'input': 0, 'output': 0, 'read': 0, 'write': 0}
                # if line_json['Stage ID'] != cur_stage:
                #     cur_metrics, cur_stage = {'input': 0, 'output': 0, 'read': 0, 'write': 0}, line_json['Stage ID']
                try:
                    all_stage_info[cur_stage]['input'] += line_json['Task Metrics']['Input Metrics']['Bytes Read']
                    all_stage_info[cur_stage]['output'] += line_json['Task Metrics']['Output Metrics']['Bytes Written']
                    all_stage_info[cur_stage]['read'] += (line_json['Task Metrics']['Shuffle Read Metrics']['Remote Bytes Read'] +
                                            line_json['Task Metrics']['Shuffle Read Metrics']['Local Bytes Read'])
                    all_stage_info[cur_stage]['write'] += line_json['Task Metrics']['Shuffle Write Metrics']['Shuffle Bytes Written']
                except:
                    print('metrics key error')
                    #continue,直接去掉这种
                    break

            if line_json['Event'] == 'SparkListenerEnvironmentUpdate':
                spark_props = line_json['Spark Properties']
                conf_list = []
                for conf_name in spark_conf_names:
                    conf_list.append(conf_name + '=' + spark_props[conf_name])
                one_data['SparkParameters'] = conf_list
            if line_json['Event'] == 'SparkListenerApplicationStart':
                start_timestamp = line_json['Timestamp']
                one_data['AppName'] = line_json['App Name']
                one_data['AppId'] = line_json['App ID']
            if line_json['Event'] == 'SparkListenerApplicationEnd':
                end_timestamp = line_json['Timestamp']

            # 按stage获取执行时间，shuffle read/write、input/output
            if line_json['Event'] == 'SparkListenerStageCompleted':
                # stage_id = 'dag_stage_' + str(line_json['Stage Info']['Stage ID'])
                try:
                    stage_id = line_json['Stage Info']['Stage ID']
                    # all_stage_info[stage_id] = cur_metrics
                    stage_start = line_json['Stage Info']['Submission Time']
                    stage_end = line_json['Stage Info']['Completion Time']
                    all_stage_info[stage_id]['duration'] = int(stage_end) - int(stage_start)
                except:
                    print('stage duration key error')
                    continue
        AppName = ''
        if end_timestamp and start_timestamp:
            if one_data['AppName'] == 'LinerRegressionApp Example':
                AppName = 'LinearRegression'
                one_data['WorkloadConf'] = env_list_all['LinearRegression']
            elif one_data['AppName'] == 'Spark ShortestPath Application':
                AppName = 'ShortestPaths'
                one_data['WorkloadConf'] = env_list_all['ShortestPaths']
            elif one_data['AppName'] == 'TeraSort':
                AppName = 'Terasort'
                one_data['WorkloadConf'] = env_list_all['Terasort']
            elif one_data['AppName'] == 'Spark StronglyConnectedComponent Application':
                AppName = 'StronglyConnectedComponent'
                one_data['WorkloadConf'] = env_list_all['StronglyConnectedComponent']
            else:
                for i in range(len(workload)):
                    if workload[i] in one_data['AppName']:
                        AppName = workload[i]
                        one_data['WorkloadConf'] = env_list_all[workload[i]]
                        break
            duration = int(end_timestamp) - int(start_timestamp)
            one_data['Duration'] = duration
            # if duration < 43000 or duration > 53000:
            # print(duration)
            # continue
        else:
            print("出错，未能获取运行时长")
            continue
        if AppName in workload_stage_count_list:
            workload_stage_count_list[AppName].append(len(all_stage_info))
        else:
            workload_stage_count_list[AppName] = [len(all_stage_info)]

        one_data['StageInfo'] = all_stage_info
        dataset_file.write(json.dumps(one_data, sort_keys=True) + '\n')
        count += 1
    print('samples count: ' + str(count))

if __name__ == '__main__':

    gen_data(history_dir=args.history_dir,
             dataset_path=args.dataset_path)


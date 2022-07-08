import os
import random
import torch
from data_process_text import load_all_code_ids
from skopt.space import Space
import numpy as np
import time
import run_action
import json


def read_log(workload):
    path = "log/" + workload
    log = open(path, 'r', encoding='utf-8')
    all_stage_info = {}
    for line in log:
        try:
            line_json = json.loads(line)
        except:
            print('json错误')
            continue
        # shuffle read/write、input/output
        if line_json['Event'] == 'SparkListenerTaskEnd':
            cur_stage = line_json['Stage ID']
            # new stage
            if line_json['Stage ID'] not in all_stage_info:
                all_stage_info[cur_stage] = [0, 0, 0, 0]
            # if line_json['Stage ID'] != cur_stage:
            #     cur_metrics, cur_stage = {'input': 0, 'output': 0, 'read': 0, 'write': 0}, line_json['Stage ID']
            try:
                all_stage_info[cur_stage][0] += line_json['Task Metrics']['Input Metrics']['Bytes Read']
                all_stage_info[cur_stage][1] += line_json['Task Metrics']['Output Metrics']['Bytes Written']
                all_stage_info[cur_stage][2] += (
                        line_json['Task Metrics']['Shuffle Read Metrics']['Remote Bytes Read'] +
                        line_json['Task Metrics']['Shuffle Read Metrics']['Local Bytes Read'])
                all_stage_info[cur_stage][3] += line_json['Task Metrics']['Shuffle Write Metrics'][
                    'Shuffle Bytes Written']
            except:
                print('metrics key error')
                break
    return len(all_stage_info.values()), list(all_stage_info.values()), list(all_stage_info.keys())


start_time = time.time()
result_sizes = [0.2, 0.5, 1, 2, 4]
workloads = ['ConnectedComponent', 'PageRank', 'ShortestPaths', 'StronglyConnectedComponent',
             'PregelOperation', 'LabelPropagation', 'TriangleCount', 'SVDPlusPlus']

app_names = ['Spark ConnectedComponent Application', 'Spark PageRank Application',
             'Spark ShortestPath Application', 'Spark StronglyConnectedComponent Application',
             'Spark PregelOperation Application', 'Spark LabelPropagation Application',
             'Spark TriangleCount Application', 'Spark SVDPlusPlus Application']
n = 7
# DecisionTree LinearRegression PCA LogisticRegression KMeans SVM Terasort ConnectedComponent PageRank PregelOperation ShortestPaths SVDPlusPlus
workload = workloads[n]
# DecisionTree classification Example / LinerRegressionApp Example / Spark PCA Example / LogisticRegressionApp Example /
# Spark KMeans Example / TeraSort / Spark ConnectedComponent Application / Spark PageRank Application /Spark PregelOperation Application
app_name = app_names[n]

# workload_feature = [250000000, 20, 3, 0]
# stage_feature = [[65536, 0, 0, 0], [104926000000, 0, 0, 0], [104926000000, 0, 0, 0], [0, 0, 2205940, 0],
#                  [104926000000, 0, 0, 0],
#                  [0, 0, 127969920, 0], [34444186904, 0, 0, 255939840], [0, 0, 255939840, 0],
#                  [34444186904, 0, 0, 508403220], [0, 0, 508403220, 0],
#                  [34444186904, 0, 0, 1013337780], [0, 0, 1013337780, 0], [34444186904, 0, 0, 2023206900],
#                  [0, 0, 2023206900, 0], [104926000000, 0, 0, 0], [104926000000, 0, 0, 0]]
# stage_count = 16
# row cols itrs partition
workload_features = {
    "DecisionTree": [250000000, 20, 3, 0],
    "LinearRegression": [600000000, 10, 3, 10],
    "LogisticRegression": [300000000, 20, 3, 10],
    "PCA": [5000000, 1000, 0, 10],
    "KMeans": [300000000, 20, 2, 2],
    "SVM": [55000000, 100, 3, 10],
    "Terasort": [800000000, 0, 3, 16],
    "ConnectedComponent": [3000000, 0, 0, 30],  # 50 10 100w 1.9
    "PageRank": [5000000, 0, 5, 20],
    "PregelOperation": [3000000, 0, 0, 60],
    "ShortestPaths": [5000000, 0, 0, 50],
    "SVDPlusPlus": [100000, 0, 1, 10],
    "StronglyConnectedComponent": [150000, 0, 0, 20],
    "TriangleCount": [250000, 0, 0, 60],
    "LabelPropagation": [2500, 0, 0, 50]
}

workload_feature = workload_features[workload]

stage_count, stage_feature, stage_ids = read_log(workload)


net_config = [8, 16, 2.9, 16, 2666, 10000]

dimension = [(1, 8), (1, 8), (1, 8), (0, 4), (1, 8), (1, 4), (1, 8), (2, 4), (1, 4), (1, 9), (1, 9),
             (1, 4), (0, 1), (1, 8), (0, 1)]
sampe_count = 25
workload_dict = {
    "Spark ConnectedComponent Application": "CC",
    "DecisionTree classification Example": "DT",
    "Spark KMeans Example": "KM",
    "LinerRegressionApp Example": "LiR",
    "LogisticRegressionApp Example": "LoR",
    "Spark LabelPropagation Application": "LP",
    "MFApp Example": "MF",
    "Spark PCA Example": "PCA",
    "Spark PregelOperation Application": "PO",
    "Spark PageRank Application": "PR",
    "Spark StronglyConnectedComponent Application": "SCC",
    "Spark ShortestPath Application": "SP",
    "Spark SVDPlusPlus Application": "SVD",
    "SVM Classifier Example": "SVM",
    "Spark TriangleCount Application": "TC",
    "TeraSort": "TS"
}


def random_x(params, stage_id):
    x = list(params)
    x[3] = result_sizes[x[3]]
    # x[7] = (16 - x[6]) * 128
    x[7] = x[7] * 512
    x[8] = x[8] * 64
    x[9] = float(x[9]) / 10
    x[10] = float(x[10]) / 10
    x[11] = x[11] * 32
    x[13] = x[13] * 32
    x.extend(workload_feature)
    # x.extend(stage_feature[stage_id])
    # x.extend(net_config)
    return x


if __name__ == '__main__':
    model = 'cc'
    space = Space(dimension)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # random.seed(2000)
    torch.manual_seed(2000)

    all_code_dict = load_all_code_ids()
    ws2nodes = torch.load('dag_data/ws2nodes.pth')
    ws2adj = torch.load('dag_data/ws2adj.pth')
    scaler = torch.load('model/scaler_' + model + '.pt')

    model1 = torch.load('model/' + model + '.pt', map_location='cpu')
    w = workload_dict[app_name]

    workload_pred_min = 1e9
    for k in range(sampe_count):
        # 15 params + workload data amount + stage's input\output\read\write
        X = []
        code_vec = []
        nodes_vec = []
        adj_vec = []

        params = space.rvs(random_state=2021 - 10 * k)[0]

        for i in stage_ids:
            x = np.array(random_x(params, stage_id=i))
            X.append(x)
            try:
                code_vec.append(all_code_dict[w][str(i)])
            except:
                code_vec.append([0 for _ in range(1000)])

            try:
                nodes_vec.append(ws2nodes[w][str(i)])
                adj_vec.append(ws2adj[w][str(i)])
            except:
                nodes_vec.append(np.zeros(64, dtype=float))
                adj_vec.append(np.zeros([64, 64], dtype=float))
        CODE = np.array(code_vec)
        nodes_vec = np.stack(nodes_vec)
        adj_vec = np.stack(adj_vec)
        X = np.array(X)

        # predict
        X = scaler.transform(X)
        X = torch.FloatTensor(X)
        CODE = torch.LongTensor(CODE)
        nodes = torch.FloatTensor(nodes_vec)
        nodes = nodes.unsqueeze(-1)
        NODES = torch.zeros(nodes.shape[0], nodes.shape[1], 64).scatter_(2, nodes.long(), 1)
        ADJS = torch.FloatTensor(adj_vec)

        workload_pred = model1(X, CODE, NODES, ADJS)
        workload_pred = workload_pred.cpu().sum()
        if workload_pred < workload_pred_min:
            workload_pred_min = workload_pred
            x_run = params
        # print(params, workload_pred)
    end_time = time.time()
    print(end_time - start_time)

    x_run1 = list(x_run)
    x_run1[0] = x_run[5]
    x_run1[1] = x_run[4]
    x_run1[2] = x_run[9]
    x_run1[3] = x_run[6]
    x_run1[4] = x_run[0]
    x_run1[5] = x_run[1]
    x_run1[6] = x_run[2]
    x_run1[7] = x_run[3]
    x_run1[8] = x_run[7]
    x_run1[9] = x_run[8]

    # Actual running time
    print(x_run1, workload_pred_min)
    run_action.run_bench(workload, x_run1)
    duration = run_action.get_rs(x_run1)
    print(duration)

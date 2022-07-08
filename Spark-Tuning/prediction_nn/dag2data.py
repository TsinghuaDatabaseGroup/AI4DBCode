#!/usr/bin/env python
# encoding: utf-8  

""" 
@author: zzz_jq
@contact: zhuangjq@stu.xmu.edu.cn
@software: PyCharm 
@file: dag2data.py 
@create: 2021/5/22 10:25 
"""

import os
import json
import torch
import numpy as np
from torch_geometric.data import Data
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="get graph of dag")

    parser.add_argument('dag_logs', type=str, help='dir of example log of each workload')

    return parser.parse_args()

args = parse_args()

dag_logs = args.dag_logs


def read_dag(dags):
    dags = sorted(dags.items(), key=lambda x: int(x[0].split('_')[2]))
    processed_dags = {}
    name_all = []
    for stage_id, dag in dags:
        processed_dags[stage_id] = []
        for node in dag:
            p_node = {}
            p_node['id'] = node['rdd_id']
            p_node['parent_ids'] = node['parent_ids']
            scope = node['scope']
            if scope is None or scope is '':
                p_node['name'] = ''
            else:
                p_node['name'] = scope['name']
                name_all.append(scope['name'])
            processed_dags[stage_id].append(p_node)
    return processed_dags


def update_child(dags):
    for stage_id, dag in dags.items():
        for node_id, node in dag.items():
            if len(node['parent_ids']) > 0:
                for parent_id in node['parent_ids']:
                    if parent_id not in dag.keys():
                        node['parent_ids'] = []
                        continue
                    if 'child_ids' not in dag[parent_id].keys():
                        dag[parent_id]['child_ids'] = []
                    dag[parent_id]['child_ids'].append(node_id)


def traverse_all_dag(all_dags, word_count):
    dags = read_dag(all_dags)
    for stage_id, dag in dags.items():
        for node_id, node in dag.items():
            if node['name'] != '':
                word_count[node['name']] = word_count.get(node['name'], 0) + 1


def build_vocab():
    raw_file_list = os.listdir(dag_logs)
    word_count = {}
    for f in raw_file_list:
        raw_file = open(dag_logs + f, 'r', encoding='utf-8')
        line = raw_file.readlines()[0]
        line_json = json.loads(line)
        all_dags = line_json['dags']
        traverse_all_dag(all_dags, word_count)
    word_count = list(word_count.items())
    word_count.sort(key=lambda k: k[1], reverse=True)
    write = open('dag_data/vocab', 'w', encoding='utf-8')
    for word_pair in word_count:
        write.write(word_pair[0] + '\t' + str(word_pair[1]) + '\n')
    write.close()


def get_w2i():
    vocab_file = open('dag_data/vocab', encoding='utf-8')
    w2i = {}
    i = 2
    for pair in vocab_file:
        v = pair.split('\t')[0]
        w2i[v] = i
        i += 1
    return w2i


def dag2data(dag, w2i):
    x = []
    src_idx = []
    tgt_idx = []
    node_id2index = {}
    for idx, node in enumerate(dag):
        node_id2index[node['id']] = idx
    for idx, node in enumerate(dag):
        if node['name'] not in w2i.keys():
            continue
        x.append(w2i[node['name']])
        if len(node['parent_ids']) == 0:
            continue
        else:
            for parent_id in node['parent_ids']:
                if parent_id not in node_id2index.keys():
                    continue
                src_idx.append(node_id2index[parent_id])
                tgt_idx.append(idx)
    print(len(x))
    x = torch.LongTensor(x).unsqueeze(1)
    edge_index = torch.tensor([src_idx, tgt_idx], dtype=torch.long)
    y = torch.FloatTensor(0)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data


def dag2matrix(dag, w2i):
    nodes = np.zeros(64, dtype=float)
    node_id2index = {}
    for idx, node in enumerate(dag):
        node_id2index[node['id']] = idx
    adj = np.zeros([64, 64], dtype=float)
    for idx, node in enumerate(dag):
        if node['name'] not in w2i.keys() or idx >= 64:
            continue
        nodes[idx] = w2i[node['name']]
        if len(node['parent_ids']) == 0:
            continue
        else:
            for parent_id in node['parent_ids']:
                if parent_id not in node_id2index.keys():
                    continue
                if idx >= 64 or node_id2index[parent_id] >= 64:
                    continue
                adj[node_id2index[parent_id], idx] = 1
    return nodes, adj


def build_graph_data():
    raw_file_list = os.listdir(dag_logs)
    ws2g = {}
    w2i = get_w2i()
    for f in raw_file_list:
        raw_file = open(dag_logs + f, 'r', encoding='utf-8')
        line = raw_file.readlines()[0]
        line_json = json.loads(line)
        all_dags = line_json['dags']
        all_dags = read_dag(all_dags)
        ws2g[f] = {}
        for stage_id, dag in all_dags.items():
            graph_data = dag2data(dag, w2i)
            s_id = stage_id.split('_')[0]
            ws2g[f][s_id] = graph_data
    torch.save(ws2g, 'dag_data/ws2g.pth')


def build_matrix_data():
    raw_file_list = os.listdir(dag_logs)
    ws2nodes, ws2adj = {}, {}
    w2i = get_w2i()
    for f in raw_file_list:
        raw_file = open(dag_logs + f, 'r', encoding='utf-8')
        line = raw_file.readlines()[0]
        line_json = json.loads(line)
        all_dags = line_json['dags']
        all_dags = read_dag(all_dags)
        ws2nodes[f], ws2adj[f] = {}, {}
        for stage_id, dag in all_dags.items():
            nodes, adj = dag2matrix(dag, w2i)
            s_id = stage_id.split('_')[-1]
            ws2nodes[f][s_id] = nodes
            ws2adj[f][s_id] = adj
    torch.save(ws2nodes, 'dag_data/ws2nodes.pth')
    torch.save(ws2adj, 'dag_data/ws2adj.pth')



if __name__ == '__main__':
    # build_vocab()
    # build_graph_data()
    build_matrix_data()


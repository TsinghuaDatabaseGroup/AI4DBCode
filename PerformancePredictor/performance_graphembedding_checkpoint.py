#!/usr/bin/env python
# coding: utf-8

from __future__ import division
from __future__ import print_function

import json
import os
import configparser
import psycopg2
import pymysql
import pymysql.cursors as pycursor

import time
import glob

cur_path = os.path.abspath('.')
data_path = os.path.join(cur_path,"pmodel_data","job")

edge_dim = 100000 # upper bound of edges
node_dim = 1000 # upper bound of nodes (at most 20 or more operators within a query in average)

'''
class DataType(IntEnum):
    Aggregate = 0
    NestedLoop = 1
    IndexScan = 2
'''
'''
argus = { "mysql": {
    "host": "166.111.121.62",
    "password": "db10204",
    "port": 3306,
    "user": "feng"},
    "postgresql": {
            "host": "166.111.121.62",
            "password": "db10204",
            "port": 5433,
            "user": "postgres"}}
argus["postgresql"]["host"]
'''
# Database -> details in Database.py
from Database import DictParser
from Database import Database

oid = 0 # operator number
min_timestamp = -1 # minimum timestamp of a graph

# extract_plan & generate_graph & add_accross_plan_relations
from extract_and_generate import extract_plan
from extract_and_generate import generate_graph
from extract_and_generate import add_across_plan_relations
from extract_and_generate import output_file

cf = DictParser()
cf.read("config.ini", encoding="utf-8")
config_dict = cf.read_dict()

# db = Database("mysql")
# print(db.fetch_knob())

# Step-0 (workload split): split the workloads into multiple concurrent queries at different time ("sample-plan-x")

'''
# Step-1 (generate (merged) workload graph): generate workload graph from the historical workloads. The merge algorithm is optional.     
start_time = time.time()
num_graphs = 3000
# notation: oid may be unused.
for wid in range(num_graphs):
    st = time.time()
    vmatrix, ematrix, mergematrix, oid, min_timestamp = generate_graph(wid, data_path)
    # optional: merge
    # vmatrix, ematrix = merge.mergegraph_main(mergematrix, ematrix, vmatrix)
    print("[graph {}]".format(wid), "time:{}; #-vertex:{}, #-edge:{}".format(time.time() - st, len(vmatrix), len(ematrix)))

    with open( os.path.join(data_path,"graph", "sample-plan-" + str(wid) + ".content"), "w") as wf:
       for v in vmatrix:
           wf.write(str(v[0]) + "\t" + str(v[1]) + "\t" + str(v[2]) + "\t" + str(v[3]) + "\t" + str(v[4]) + "\n")
    with open( os.path.join(data_path, "graph" , "sample-plan-" + str(wid) + ".cites"), "w") as wf:
       for e in ematrix:
           wf.write(str(e[0]) + "\t" + str(e[1]) + "\t" + str(e[2]) + "\n")

end_time = time.time()
print("Total Time:{}".format(end_time - start_time))
'''
# output_file()

# Step-2: load graph data
graphs = glob.glob("./pmodel_data/job/graph/sample-plan-*")
num_graphs = int(len(graphs)/2) # (content, cites)
print("[Generated Graph]", num_graphs)

# Graph Embedding Algorithm
import numpy as np
import torch
import torch.nn.functional as F

x=np.asarray([[1,2], [3, 4]])
X=torch.Tensor(x)
print(X.shape)
pad_dims = (1, 3)
X=F.pad(X,pad_dims,"constant")
print(X)
print(X.shape[0])

# GCN model -> details in GCN.py
from GCN import arguments
args = arguments()

from pathlib import Path
print(Path().resolve())
from GCN import GCN

import time
import numpy as np
# dataloader -> details in dataloader.py
from dataloader import accuracy
from dataloader import load_data
from dataloader import load_data_from_matrix

import torch.nn.functional as F
import torch.optim as optim

def train(epoch, labels, features, adj, idx_train):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    # print(output[idx_train])
    # print("output = !",output,"labels = !", labels)
    loss_train = F.mse_loss(output[idx_train], labels[idx_train])
    
    # loss_train = nn.CrossEntropyLoss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)
        # transfer output to ms
        # output = output * 1000
    # https://www.cnblogs.com/52dxer/p/13793911.html
    loss_val = F.mse_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    
    return round(loss_train.item(), 4)

def test(labels, idx_test):
    model.eval()
    output = model(features, adj)
    # transfer output to ms
    # output = output * 1000
    loss_test = F.mse_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))

# Step-3 (runtime prediction):
feature_num = 3
num_graphs = 10
# graphs = glob.glob("./pmodel_data/job/sample-plan-*")
# num_graphs = len(graphs)
iteration_num = int(round(0.8 * num_graphs, 0)) 
print("[training samples]:{}".format(iteration_num))

model = GCN(nfeat=feature_num,
            nhid=args.hidden,
            nclass=node_dim, 
            dropout=args.dropout)    

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


import random

for wid in range(iteration_num):

    gid = random.randint(0, iteration_num)

    print("[graph {}]".format(gid))
    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data(path = os.path.join(data_path,"graph"), dataset = "sample-plan-" + str(gid))
    # print(adj.shape)
    
    # Model Training
    ok_times = 0
    t_total = time.time()
    labels = labels * 10
    for epoch in range(args.epochs):
        # print(features.shape, adj.shape)
        loss_train = train(epoch, labels, features, adj, idx_train)
        if loss_train < 0.002:
            ok_times += 1
        if ok_times >= 20:
            break    
    
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Model Validation
    test(labels, idx_test)

for wid in range(iteration_num, num_graphs):
    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data(path = os.path.join(data_path, "graph/"), dataset = "sample-plan-" + str(wid))
    
    # Model Testing
    t_total = time.time()
    test(labels, idx_test)
    print("Testing Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

exit()


# assume graph_num >> come_num
graph_num = 4
come_num = 1

# train model on a big graph composed of graph_num samples
min_timestamp = -1
vmatrix = []
ematrix = [] 
conflict_operators = {}

for wid in range(graph_num):
    
    with open( os.path.join(data_path, "sample-plan-" + str(wid) + ".txt"), "r") as f:

        for sample in f.readlines():
            sample = json.loads(sample)
            
            start_time, node_matrix, edge_matrix, conflict_operators, _ , min_timestamp = extract_plan(sample, conflict_operators, oid, min_timestamp)
            
            vmatrix = vmatrix + node_matrix
            ematrix = ematrix + edge_matrix

db = Database("mysql")
knobs = db.fetch_knob()               
ematrix = add_across_plan_relations(conflict_operators, knobs, ematrix)

# TODO more features, more complicated model
model = GCN(nfeat=feature_num,
            nhid=args.hidden,
            nclass=node_dim, 
            dropout=args.dropout)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

adj, features, labels, idx_train, idx_val, idx_test = load_data_from_matrix(np.array(vmatrix, dtype=np.float32), np.array(ematrix, dtype=np.float32))

ok_times = 0
for epoch in range(args.epochs):
    # print(features.shape, adj.shape)
    loss_train = train(epoch, labels, features, adj, idx_train)
    if loss_train < 0.002:
        ok_times += 1
    if ok_times >= 20:
        break 

test(labels, idx_test)

def predict(labels, features, adj, dh):
    model.eval()
    output = model(features, adj, dh)
    loss_test = F.mse_loss(output, labels)
    acc_test = accuracy(output, labels)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))

import bisect

k = 20
new_e = []
conflict_operators = {}
phi = []

for wid in range(graph_num, graph_num + come_num):

    with open(os.path.join(data_path, "sample-plan-" + str(wid) + ".txt"), "r") as f:

        # new query come
        for sample in f.readlines():

            # updategraph-add
            sample = json.loads(sample)

            start_time, node_matrix, edge_matrix, conflict_operators, _, min_timestamp = extract_plan(sample, conflict_operators, oid, min_timestamp)

            vmatrix = vmatrix + node_matrix
            new_e = new_e + edge_matrix

            knobs = db.fetch_knob()

            new_e = add_across_plan_relations(conflict_operators, knobs, new_e)

            # incremental prediction
            dadj, dfeatures, dlabels, _, _, _ = load_data_from_matrix(np.array(vmatrix, dtype=np.float32), np.array(new_e, dtype=np.float32))

            model.eval()
            dh = model(dfeatures, dadj, None, True)

            predict(dlabels, dfeatures, adj, dh)

            for node in node_matrix:
                bisect.insort(phi, [node[-2] + node[-1], node[0]])

            # updategraph-remove
            num = bisect.bisect(phi, [start_time, -1])
            if num > k:
                rmv_phi = [e[1] for e in phi[:num]]
                phi = phi[num:]
                vmatrix = [v for v in vmatrix if v[0] not in rmv_phi]
                new_e = [e for e in new_e if e[0] not in rmv_phi and e[1] not in rmv_phi]
                for table in conflict_operators:
                    conflict_operators[table] = [v for v in conflict_operators[table] if v[0] not in rmv_phi]

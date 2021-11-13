#!/usr/bin/env python
# coding: utf-8

# In[1]:
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import os

import configparser
import psycopg2
import pymysql
import pymysql.cursors as pycursor
import numpy as np

import time
import glob


# # 1. Generate Workload Dataset

# In[2]:


cur_path = os.path.abspath('.')
data_path = cur_path + '/pmodel_data/job/'

edge_dim = 25000 # upper bound of edges
node_dim = 300 # upper bound of nodes

'''
class DataType(IntEnum):
    Aggregate = 0
    NestedLoop = 1
    IndexScan = 2
'''
mp_optype = {'Aggregate': 0, 'Nested Loop': 1, 'Index Scan': 2, 'Hash Join': 3, 'Seq Scan': 4, 'Hash': 5, 'Update': 6} # operator types in the queries

oid = 0 # operator number

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


# In[3]:


# obtain and normalize configuration knobs

class DictParser(configparser.ConfigParser):
    def read_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(d[k])
        return d

cf = DictParser()
cf.read("config.ini", encoding="utf-8")
config_dict = cf.read_dict()


def parse_knob_config():
    _knob_config = config_dict["knob_config"]
    for key in _knob_config:
        _knob_config[key] = json.loads(str(_knob_config[key]).replace("\'", "\""))
    return _knob_config


class Database:
    def __init__(self, server_name='postgresql'):
        
        knob_config = parse_knob_config()
        self.knob_names = [knob for knob in knob_config]
        self.knob_config = knob_config
        self.server_name = server_name
        
        # print("knob_names:", self.knob_names)
        
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            sql = "SELECT count FROM INFORMATION_SCHEMA.INNODB_METRICS where status='enabled'"
            cursor.execute(sql)
            result = cursor.fetchall()

            self.internal_metric_num = len(result)
            cursor.close()
            conn.close()
        except Exception as err:
            print("execute sql error:", err)

    def _get_conn(self):
        if self.server_name == 'mysql':
            sucess = 0
            conn = -1
            count = 0
            while not sucess and count < 3:
                try:
                    conn = pymysql.connect(host="166.111.121.62",
                                           port=3306,
                                           user="feng",
                                           password="db10204",
                                           db='INFORMATION_SCHEMA',
                                           connect_timeout=36000,
                                           cursorclass=pycursor.DictCursor)

                    sucess = 1
                except Exception as result:
                    count += 1
                    time.sleep(10)
            if conn == -1:
                raise Exception
                
            return conn
            
        elif self.server_name == 'postgresql':
            sucess = 0
            conn = -1
            count = 0
            while not sucess and count < 3:
                try:
                    db_name = "INFORMATION_SCHEMA" # zxn Modified.
                    conn = psycopg2.connect(database="INFORMATION_SCHEMA", user="lixizhang", password="xi10261026zhang", host="166.111.5.177", port="5433")
                    sucess = 1
                except Exception as result:
                    count += 1
                    time.sleep(10)
            if conn == -1:
                raise Exception
            return conn

        else:
            print('数据库连接不上...')
            return

    def fetch_knob(self):
        state_list = np.append([], [])
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            sql = "select"
            for i, knob in enumerate(self.knob_names):
                sql = sql + ' @@' + knob

                if i < len(self.knob_names) - 1:
                    sql = sql + ', '

            # state metrics
            cursor.execute(sql)
            result = cursor.fetchall()
            
            for i in range(len(self.knob_names)):
                value = result[0]["@@%s" % self.knob_names[i]] if result[0]["@@%s" % self.knob_names[i]]!=0 else self.knob_config[self.knob_names[i]]["max_value"] # not limit if value equals 0
                
                # print(value, self.knob_config[self.knob_names[i]]["max_value"], self.knob_config[self.knob_names[i]]["min_value"])
                state_list = np.append(state_list, value / (self.knob_config[self.knob_names[i]]["max_value"] - self.knob_config[self.knob_names[i]]["min_value"]))
            cursor.close()
            conn.close()
        except Exception as error:
            print("fetch_knob Error:", error)
        
        return state_list

# db = Database("mysql")
# print(db.fetch_knob())


# In[4]:


# actual runtime:  actuall executed (training data) / estimated by our model
# operators in the same plan can have data conflicts (parallel)

def compute_cost(node):
    return float(node["Total Cost"]) - float(node["Startup Cost"]) 

def compute_time(node):
    # return float(node["Actual Total Time"]) - float(node["Actual Startup Time"]) 
    return float(node["Actual Total Time"]) # mechanism within pg
    
def get_used_tables(node):

    tables = []

    stack = [node]
    while stack != []:
        parent = stack.pop(0)
        
        if "Relation Name" in parent:
            tables.append(parent["Relation Name"])
        
        if "Plans" in parent:
            for n in parent["Plans"]:
                stack.append(n)

    return  tables


def extract_plan(sample, conflict_operators):
    global mp_optype, oid
    # function: extract SQL feature
    # return: start_time, node feature, edge feature
    
    plan = sample["plan"]
    while isinstance(plan, list):
        plan = plan[0]
    # Features: print(plan.keys()) 
        # start time = plan["start_time"]
        # node feature = [Node Type, Total Cost:: Actual Total Time]
        # node label = [Actual Startup Time, Actual Total Time]

    plan = plan["Plan"] # root node
    node_matrix = []
    edge_matrix = []
    node_merge_matrix = []

    # add oid for each operator
    stack = [plan]
    while stack != []:
        parent = stack.pop(0)
        parent["oid"] = oid
        oid = oid + 1
        
        if "Plans" in parent:
            for node in parent["Plans"]:
                stack.append(node)    
    
    stack = [plan]
    while stack != []:
        parent = stack.pop(0)
        run_cost = compute_cost(parent)
        run_time = compute_time(parent)
        # print(parent["Actual Total Time"], parent["Actual Startup Time"], run_time)
        
        if parent["Node Type"] not in mp_optype:
            mp_optype[parent["Node Type"]] = len(mp_optype)
        
        tables = get_used_tables(parent)
        # print("[tables]", tables)
        
        operator_info = [parent["oid"], parent["Startup Cost"], parent["Total Cost"]]
        
        for table in tables:
            if table not in conflict_operators:
                conflict_operators[table] = [operator_info]
            else:
                conflict_operators[table].append(operator_info)
        
                
        node_feature = [parent["oid"], mp_optype[parent["Node Type"]], run_cost, float(parent["Actual Startup Time"]), run_time]
        
        node_matrix = [node_feature] + node_matrix

        node_merge_feature = [parent["oid"], parent["Startup Cost"], parent["Total Cost"], mp_optype[parent["Node Type"]], run_cost, float(parent["Actual Startup Time"]), run_time]
        node_merge_matrix = [node_merge_feature]  + node_merge_matrix
        # [id?, l, r, ....]
        
        if "Plans" in parent:
            for node in parent["Plans"]:
                stack.append(node)
                edge_matrix = [[node["oid"], parent["oid"], 1]] + edge_matrix

    # node: 18 * featuers
    # edge: 18 * 18

    return float(sample["start_time"]), node_matrix, edge_matrix, conflict_operators, node_merge_matrix


def overlap(node_i, node_j):
    
    if (node_j[1] < node_i[2] and node_i[2] < node_j[2]):
        
        return (node_i[2] - node_j[1]) / (node_j[2] - min(node_i[1], node_j[1]))
    
    elif (node_i[1] < node_j[2] and node_j[2] < node_i[2]):
        
        return (node_j[2] - node_i[1]) / (node_i[2] - min(node_i[1], node_j[1]))
    
    else:
        return 0

def add_across_plan_relations(conflict_operators, knobs, ematrix):
    
    data_weight = 0.1
    for knob in knobs:
        data_weight *= knob
    # print(conflict_operators)

    data_matrix = ematrix
    # add relations [rw/ww, rr, config]
    for table in conflict_operators:
        for i in range(len(conflict_operators[table])):
            for j in range(i+1, len(conflict_operators[table])):

                node_i = conflict_operators[table][i]
                node_j = conflict_operators[table][j]
                
                time_overlap = overlap(node_i, node_j)
                if time_overlap:
                    data_matrix.append([node_i[0], node_j[0], -data_weight * time_overlap])
                    data_matrix.append([node_j[0], node_i[0], -data_weight * time_overlap])
                '''
                if overlap(i, j) and ("rw" or "ww"):
                    ematrix = ematrix + [[conflict_operators[table][i], conflict_operators[table][j], data_weight * time_overlap]]
                    ematrix = ematrix + [[conflict_operators[table][j], conflict_operators[table][i], data_weight * time_overlap]]
                '''

    # print(data_matrix[:-1])

    return data_matrix

import merge

def generate_graph(wid, path = data_path):
    global oid
    # fuction
    # return
    # todo: timestamp
    
    vmatrix = []
    ematrix = []
    mergematrix = []
    conflict_operators = {}

    oid = 0
    with open(path + "sample-plan-" + str(wid) + ".txt", "r") as f:
        
        # vertex: operators
        # edge: child-parent relations
        for sample in f.readlines():
            
            sample = json.loads(sample)
            
            # Step 1: read (operators, parent-child edges) in separate plans
            start_time, node_matrix, edge_matrix, conflict_operators, node_merge_matrix = extract_plan(sample, conflict_operators)

            mergematrix = mergematrix + node_merge_matrix
            vmatrix = vmatrix + node_matrix
            ematrix = ematrix + edge_matrix

    # ZXN TEMP Modified BEGIN
        # Step 2: read related knobs
        db = Database("mysql")
        knobs = db.fetch_knob()

        # Step 3: add relations across queries
        ematrix2 = add_across_plan_relations(conflict_operators, knobs, ematrix)
        # print(ematrix2)
        # edge: data relations based on (access tables, related knob values)
        vmatrix, ematrix = merge.mergegraph_main(mergematrix, ematrix2, vmatrix)
    # ZXN TEMP Modified ENDED
    return vmatrix, ematrix, mergematrix





# '''
# Split the workloads into multiple concurrent queries at different time ("sample-plan-x")

workloads = glob.glob("./pmodel_data/job/sample-plan-*")
num_graphs = len(workloads) # change
start_time = time.time()

# convert into (vmatrix, ematrix)
for wid in range(num_graphs):
    st = time.time()
    vmatrix, ematrix, mergematrix = generate_graph(wid)
    ''' 
    Note: ematrix (after merge) is an array of edge matrices. 
    In each edge matrix, there are at most one edge between two vertices. 
    And we use the embeddings of all the edge matrices to predict the performance. 
    '''
    print("[graph {}]".format(wid), "time:{}; #-vertex:{}, #-edge:{}".format(time.time() - st, len(vmatrix), len(ematrix)))

    # write into files
    with open( os.path.join (data_path, "merged-graph", "sample-plan-" + str(wid) + ".content"), "w") as wf:
       for v in mergematrix:
           wf.write(str(v[0]) + "\t" + str(v[1]) + "\t" + str(v[2]) + "\t" + str(v[3]) + "\t" + str(v[4]) + "\t" + str(v[5]) + "\t" + str(v[6]) + "\n")
    for i, edgematrix in enumerate(ematrix):
        with open( os.path.join (data_path, "merged-graph" , "sample-plan-" + str(wid) + "-" + str(i) + ".cites"), "w") as wf:
           for e in edgematrix:
               wf.write(str(e[0]) + "\t" + str(e[1]) + "\t" + str(e[2]) + "\n")

end_time = time.time()

print("Total Time:{}".format(end_time - start_time))
# '''

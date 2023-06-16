import itertools
import random
import numpy as np
from scipy.optimize import minimize
import sys
import os

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

import paramiko
import re
import subprocess
import numpy as np
import math

from api.services.partition.config import PartitionConfig
from api.services.partition.utils import utils
from api.services.partition.database import database
from api.services.partition.models.attention_network import AttentionNetwork
from api.services.partition.models.gnn import GNN
from api.services.partition.partition_evaluation.evaluation_model import SampleGraph, partition_evaluation_model

import pdb

sys.path.append('api/services')

enable_edge_weight = 1
enable_vertex_features = 1
enable_vertex_attention = 1
enable_taylor_decomposition = 1
enable_evaluation_model = 0
enable_model_preloading = 1

# evaluate the query latency under the selected partitioning keys
if __name__ == "__main__":

    # 生成参数
    args = PartitionConfig()
    # 生成路径
    success, msg = args.generate_paths()
    if not success:
        raise ValueError(msg)

    partitioning_keys = {'lineitem': ['l_orderkey', 'l_quantity'], 'orders': ['o_custkey', 'o_orderdate'], 'customer': ['c_custkey']}

    # Generate k-node sample graph (self.sample_vertex_matrix, self.sample_edge_matrix)
    partitioned_sample_graph = SampleGraph(args, partitioning_keys=partitioning_keys)
    partition_eval = partition_evaluation_model(args)

    # compute the embedding for the partitioned_sample_graph    
    embedding = partition_eval.embedding(partitioned_sample_graph)
    
    # estimate the query latency based on the embedding for the partitioned_sample_graph    
    latency = partition_eval.estimate_latency(embedding)
    print(latency)
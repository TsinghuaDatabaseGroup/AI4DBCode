import torch
import torch.nn as nn
import os
import re
import time
import logging
import pickle

from api.services.partition.database import database
from api.services.partition.partition_selection.selection_model import Column2Graph, partitioning_model
from api.services.partition.partition_evaluation.evaluation_model import SampleGraph, partition_evaluation_model
from api.services.partition.config import PartitionConfig


def generate_training_data():

    # 生成参数
    args = PartitionConfig()
    # 生成路径
    success, msg = args.generate_paths()
    if not success:
        raise ValueError(msg)

    current_timestamp = int(time.time())
    print("[origin database]: ", args.database)

    # Set up logging
    logging.basicConfig(filename='./logs/train_partitioning_models_{}.log'.format(current_timestamp), level=logging.INFO)

    # configure key selection model
    p_model = partitioning_model(args)
    # p_optimizer = torch.optim.SGD(p_model.gnns[0].parameters(), lr=args.partition_learning_rate)
    # loss_fn = nn.MSELoss()

    # configure evaluation model
    e_model = partition_evaluation_model(args)
    e_optimizer = torch.optim.Adam(list(e_model.gnn.parameters()) + list(e_model.fc_layer.parameters()), 
                                 lr=args.evaluation_learning_rate)
    e_criterion = nn.MSELoss()

    # obtain the default latency and throughput
    partition_keys = {}
    pattern = r'DISTRIBUTED\s+BY\s*\(\s*(.*?)\s*\)\s*;'
    with open(args.schema_path, "r") as f:
        text = f.read()
    table_defs = text.split('CREATE TABLE ')[1:]
    for table_def in table_defs:
        table_name, rest = table_def.split('(', 1)
        table_name = table_name[:-1]
        match = re.search(pattern, rest, re.IGNORECASE)
        if match:
            keys = match.group(1).split(',')
            partition_keys[table_name] = ','.join(keys)
            print(f"{table_name}: {keys}")
        else:
            partition_keys[table_name] = ""
            print(f"{table_name}: No distributed keys found --> Random Distribution")

    default_latency, default_throughput = database.execution_under_selected_keys(args, args.database, partition_keys)
    logging.info(f"Default Latency: {default_latency}, Default Throughput: {default_throughput}")
    logging.info(f"Partitioning Keys: {partition_keys}")
    print(f"Default Latency: {default_latency}, Default Throughput: {default_throughput}")

    # sample the real data into a sample database
    # database.clone_sample_data_to_database(args) # very slow! only run once
    
    logging.info(f"Sample data with sample ratio as {args.sample_ratio}")
    print(f"Sample data with sample ratio as {args.sample_ratio}")


    graph = Column2Graph(args)
    candidate_cols = list(graph.used_cols.keys())

    success_time = 0
    for epoch in range(args.training_epochs):

        # Training partitioning model        
        partitioning_keys_marks = p_model.forward(graph)

        partitioning_keys = {}
        for i in range(len(partitioning_keys_marks)):
            if partitioning_keys_marks[i] == 1:
                if graph.used_cols[candidate_cols[i]] not in partitioning_keys:
                    partitioning_keys[graph.used_cols[candidate_cols[i]]] = candidate_cols[i]
                else:
                    partitioning_keys[graph.used_cols[candidate_cols[i]]] = partitioning_keys[graph.used_cols[candidate_cols[i]]] + "," + candidate_cols[i]
         
        partitioned_sample_graph = SampleGraph(args, partitioning_keys=partitioning_keys, is_sample=True)

        embedding = e_model.embedding(partitioned_sample_graph)
        estimated_latency = e_model.estimate_latency(embedding)
        
        p_loss = -torch.mean(torch.abs(default_latency - estimated_latency)/(default_latency + 1e-10))
        p_model.compute_gradient(p_loss)

        real_latency, real_throughput = database.execution_under_selected_keys(args, args.database, partitioning_keys)

        metric1 = {'latency': real_latency, 'throughput': real_throughput}
        new_data = (partitioned_sample_graph, metric1)

        # Read the existing data and append the new data
        file_path = 'sampled_training_data.pickle'
        try:
            with open(file_path, 'rb') as f_exp_data:
                existing_data = pickle.load(f_exp_data)
        except (EOFError, FileNotFoundError):
            existing_data = []

        try:
            with open('distinct_'+file_path, 'rb') as f_exp_data:                
                existing_data_distinct = pickle.load(f_exp_data)
        except (EOFError, FileNotFoundError):
            existing_data_distinct = []

        if new_data not in existing_data_distinct:
            existing_data_distinct.append(new_data)
        existing_data.append(new_data)

        # Write the updated data back to the file
        with open(file_path, 'wb') as f_exp_data:
            pickle.dump(existing_data, f_exp_data)

        with open('distinct_'+file_path, 'wb') as f_exp_data:
            pickle.dump(existing_data, f_exp_data)


        print(f"The {epoch}th epoch: {partitioning_keys_marks}")
        time.sleep(1)


if __name__ == "__main__":
    
    generate_training_data()
import torch
import torch.nn as nn
import os
import re
import time
import logging

from api.services.partition.utils import utils
from api.services.partition.database import database
from api.services.partition.partition_selection.selection_model import Column2Graph, partitioning_model
from api.services.partition.partition_evaluation.evaluation_model import SampleGraph, partition_evaluation_model
from api.services.partition.config import PartitionConfig


def train_partitioning_models():

    # 生成参数
    args = PartitionConfig()
    # 生成路径
    success, msg = args.generate_paths()
    if not success:
        raise ValueError(msg)

    current_timestamp = int(time.time())
    args.database = f"{args.database}_{current_timestamp}"
    print(" === origin database: ", args.database)

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

    #p_loss = 0.0
    best_keys = partition_keys
    best_latency = default_latency

    if args.reload_pretrain == True:

        logging.info(f"Reload pre-trained models from {args.pretrain_model_checkpoint}")
        print(f"Reload pre-trained models from {args.pretrain_model_checkpoint}")

        p_model_path = os.path.join(args.pretrain_model_checkpoint, 'partitioning_model.pt')
        e_model_path = os.path.join(args.pretrain_model_checkpoint, 'evaluation_model.pt')

        if os.path.exists(p_model_path):
            p_model.gnns[0].load_state_dict(torch.load(p_model_path))
        if os.path.exists(e_model_path):
            e_model.gnn.load_state_dict(torch.load(e_model_path))

    success_time = 0
    default_latency = torch.tensor(default_latency, requires_grad=True)
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

        # print p_model parameters
        # for name, param in p_model.gnns[0].named_parameters():
        #     print(f"Parameter Name: {name}")
        #     print(f"Parameter Shape: {param.shape}")
        #     print(f"Parameter Values: {param}")
        #     print()

        p_model.compute_gradient(p_loss) # update the p_model weights with customized gradient computation function

        # mu = 0.001  # Update rate
        # delta_v = abs((default_latency - estimated_latency)/(default_latency + 1e-10))
        # loss_value = mu / (delta_v + 1e-6)
        # loss_tensor = torch.tensor(loss_value, requires_grad=True)

        # p_loss = -torch.mean(loss_tensor)
        # p_optimizer.zero_grad()
        # p_loss.backward(retain_graph=True)
        # p_optimizer.step()

        # Gain true performance under the selected partitioning keys
        real_latency, real_throughput = database.execution_under_selected_keys(args, args.database+"_tmp", partitioning_keys)
        real_latency = torch.tensor(real_latency, requires_grad=True)

        if real_latency < best_latency * 0.9: # reduce the effect of normal performance noise
            best_latency = real_latency.item()
            best_keys = partitioning_keys


        if real_latency < default_latency * 0.9:
        
            if success_time % 10 == 0:
                success_time = 0
                # report error if args.pretrain_model_dir does not exist
                if not os.path.exists(args.saved_model_dir):
                    raise Exception("Saved model directory does not exist!")

                current_model_dir = os.path.join(args.saved_model_dir, args.database)
                os.mkdir(current_model_dir)

                # Save models if real_latency improved
                torch.save(p_model.gnns[0].state_dict(), os.path.join(current_model_dir, 'partitioning_model.pt'))
                torch.save(e_model.gnn.state_dict(), os.path.join(current_model_dir, 'evaluation_model.pt'))

            success_time += 1

        # Training evaluation model
        td_error = e_criterion(real_latency, estimated_latency)
        # denominator = torch.abs(true_performance - estimated_latency)
        # lmls_loss = torch.mean(torch.log(1 + 1 / 2 * denominator)) # Least Mean Log Squares (LMLS)
        e_optimizer.zero_grad()
        td_error.backward(retain_graph=True)
        e_optimizer.step()

        # drop the tmp database
        database.drop_database(args, args.database+"_tmp")

        # log the training process
        logging.info(f"Epoch: {epoch}, Partitioning Loss: {p_loss}, Evaluation Loss: {td_error}")
        logging.info(f"Estimated Latency: {estimated_latency}, Real Latency: {real_latency.item()}, Real Throughput: {real_throughput}")
        logging.info(f"Partitioning Keys: {partitioning_keys}")


    logging.info(f"Best Latency: {best_latency}, Best Keys: {best_keys}")
    print(f"Best Latency: {best_latency}, Best Keys: {best_keys}")
    print('Finished Training')


if __name__ == "__main__":
    train_partitioning_models()
from api.services.partition.partition_selection.selection_model import Column2Graph, partitioning_model
import os
import torch
import torch.nn as nn

def partition_key_selection(args):

    p_model = partitioning_model(args)
    
    if args.reload_pretrain == True:
        p_model_path = os.path.join(args.pretrain_model_checkpoint, 'partitioning_model.pt')

        if os.path.exists(p_model_path):
            p_model.gnns[0].load_state_dict(torch.load(p_model_path))

    graph = Column2Graph(args)

    candidate_cols = list(graph.used_cols.keys())
    partitioning_keys_marks = p_model.forward(graph)
    partitioning_keys = {}
    for i in range(len(partitioning_keys_marks)):
        if partitioning_keys_marks[i] == 1:
            if graph.used_cols[candidate_cols[i]] not in partitioning_keys:
                partitioning_keys[graph.used_cols[candidate_cols[i]]] = [
                    candidate_cols[i]]
            else:
                partitioning_keys[graph.used_cols[candidate_cols[i]]].append(
                    candidate_cols[i])

    return partitioning_keys
    # {'lineitem': ['l_quantity', 'l_shipdate'], 'orders': ['o_orderkey', 'o_custkey'], 'customer': ['c_custkey']}

from api.services.partition.partition_selection.selection_model import Column2Graph, partitioning_model


def partition_key_selection(args):
    p_model = partitioning_model(args)

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

from src.feature_extraction.node_features import *


def plan2seq(root, alias2table):
    sequence = []
    join_conditions = []
    node, join_condition = extract_info_from_node(root, alias2table)
    if join_condition is not None:
        join_conditions += join_condition
    sequence.append(node)
    if 'Plans' in root:
        for plan in root['Plans']:
            next_sequence, next_join_conditions = plan2seq(plan, alias2table)
            sequence += next_sequence
            join_conditions += next_join_conditions
    sequence.append(None)
    return sequence, join_conditions

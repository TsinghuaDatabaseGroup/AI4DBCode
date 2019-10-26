from src.plan_encoding.encoding_predicates import *


def encode_sample(sample):
    return np.array([int(i) for i in sample])


def bitand(sample1, sample2):
    return np.minimum(sample1, sample2)


def encode_node_job(node, parameters):
    # operator + first_condition + second_condition + relation
    extra_info_num = max(parameters.column_total_num, parameters.table_total_num, parameters.index_total_num)
    operator_vec = np.array([0 for _ in range(parameters.physic_op_total_num)])

    extra_info_vec = np.array([0 for _ in range(extra_info_num)])
    condition1_vec = np.array([[0 for _ in range(parameters.condition_op_dim)] for _ in range(parameters.condition_max_num)])
    condition2_vec = np.array([[0 for _ in range(parameters.condition_op_dim)] for _ in range(parameters.condition_max_num)])
    ### Samples Starts
    sample_vec = np.array([1 for _ in range(1000)])
    ### Samples Ends
    has_condition = 0
    if node != None:
        operator = node['node_type']
        operator_idx = parameters.physic_ops_id[operator]
        operator_vec[operator_idx - 1] = 1
        if operator == 'Materialize' or operator == 'BitmapAnd' or operator == 'Result':
            pass
        elif operator == 'Sort':
            for key in node['sort_keys']:
                extra_info_inx = parameters.columns_id[key]
                extra_info_vec[extra_info_inx - 1] = 1
        elif operator == 'Hash Join' or operator == 'Merge Join' or operator == 'Nested Loop':
            condition1_vec = encode_condition(node['condition'], None, None, parameters)
        elif operator == 'Aggregate':
            for key in node['group_keys']:
                extra_info_inx = parameters.columns_id[key]
                extra_info_vec[extra_info_inx - 1] = 1
        elif operator == 'Seq Scan' or operator == 'Bitmap Heap Scan' or operator == 'Index Scan'\
                or operator == 'Bitmap Index Scan' or operator == 'Index Only Scan':
            relation_name = node['relation_name']
            index_name = node['index_name']
            if relation_name is not None:
                extra_info_inx = parameters.tables_id[relation_name]
            else:
                extra_info_inx = parameters.indexes_id[index_name]
            extra_info_vec[extra_info_inx - 1] = 1
            condition1_vec = encode_condition(node['condition_filter'], relation_name, index_name, parameters)
            condition2_vec = encode_condition(node['condition_index'], relation_name, index_name, parameters)
            if 'bitmap' in node:
                ### Samples Starts
                sample_vec = encode_sample(node['bitmap'])
                ### Samples Ends
                has_condition = 1
            if 'bitmap_filter' in node:
                ### Samples Starts
                sample_vec = bitand(encode_sample(node['bitmap_filter']), sample_vec)
                ### Samples Ends
                has_condition = 1
            if 'bitmap_index' in node:
                ### Samples Starts
                sample_vec = bitand(encode_sample(node['bitmap_index']), sample_vec)
                ### Samples Ends
                has_condition = 1

                #     print 'operator: ', operator_vec
                #     print 'extra_infos: ', extra_info_vec
    return operator_vec, extra_info_vec, condition1_vec, condition2_vec, sample_vec, has_condition

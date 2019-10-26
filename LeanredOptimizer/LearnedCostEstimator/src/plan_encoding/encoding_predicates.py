import re

import numpy as np

from src.plan_encoding.meta_info import *


def get_representation(value, word_vectors):
    if value in word_vectors:
        embedded_result = np.array(list(word_vectors[value]))
    else:
        embedded_result = np.array([0.0 for _ in range(500)])
    hash_result = np.array([0.0 for _ in range(500)])
    for t in value:
        hash_result[hash(t) % 500] = 1.0
    return np.concatenate((embedded_result, hash_result), 0)


def get_str_representation(value, column, word_vectors):
    vec = np.array([])
    count = 0
    prefix = determine_prefix(column)
    for v in value.split('%'):
        if len(v) > 0:
            if len(vec) == 0:
                vec = get_representation(prefix + v, word_vectors)
                count = 1
            else:
                new_vec = get_representation(prefix + v, word_vectors)
                vec = vec + new_vec
                count += 1
    if count > 0:
        vec /= float(count)
    return vec


def encode_condition_op(condition_op, relation_name, index_name, parameters):
    # bool_operator + left_value + compare_operator + right_value
    if condition_op is None:
        vec = [0 for _ in range(parameters.condition_op_dim)]
    elif condition_op['op_type'] == 'Bool':
        idx = parameters.bool_ops_id[condition_op['operator']]
        vec = [0 for _ in range(parameters.bool_ops_total_num)]
        vec[idx - 1] = 1
    else:
        operator = condition_op['operator']
        left_value = condition_op['left_value']
        if re.match(r'.+\..+', left_value) is None:
            if relation_name is None:
                relation_name = index_name.split(left_value)[1].strip('_')
            left_value = relation_name + '.' + left_value
        else:
            relation_name = left_value.split('.')[0]
        left_value_idx = parameters.columns_id[left_value]
        left_value_vec = [0 for _ in range(parameters.column_total_num)]
        left_value_vec[left_value_idx - 1] = 1
        right_value = condition_op['right_value']
        column_name = left_value.split('.')[1]
        if re.match(r'^[a-z][a-zA-Z0-9_]*\.[a-z][a-zA-Z0-9_]*$', right_value) is not None and right_value.split('.')[
            0] in parameters.data:
            operator_idx = parameters.compare_ops_id[operator]
            operator_vec = [0 for _ in range(parameters.compare_ops_total_num)]
            operator_vec[operator_idx - 1] = 1
            right_value_idx = parameters.columns_id[right_value]
            right_value_vec = [0]
            left_value_vec[right_value_idx - 1] = 1
        elif parameters.data[relation_name].dtypes[column_name] == 'int64' or parameters.data[relation_name].dtypes[column_name] == 'float64':
            right_value = float(right_value)
            value_max = parameters.min_max_column[relation_name][column_name]['max']
            value_min = parameters.min_max_column[relation_name][column_name]['min']
            right_value_vec = [(right_value - value_min) / (value_max - value_min)]
            operator_idx = parameters.compare_ops_id[operator]
            operator_vec = [0 for _ in range(parameters.compare_ops_total_num)]
            operator_vec[operator_idx - 1] = 1
        elif re.match(r'^__LIKE__', right_value) is not None:
            operator_idx = parameters.compare_ops_id['~~']
            operator_vec = [0 for _ in range(parameters.compare_ops_total_num)]
            operator_vec[operator_idx - 1] = 1
            right_value = right_value.strip('\'')[8:]
            right_value_vec = get_str_representation(right_value, left_value, parameters.word_vectors).tolist()
        elif re.match(r'^__NOTLIKE__', right_value) is not None:
            operator_idx = parameters.compare_ops_id['!~~']
            operator_vec = [0 for _ in range(parameters.compare_ops_total_num)]
            operator_vec[operator_idx - 1] = 1
            right_value = right_value.strip('\'')[11:]
            right_value_vec = get_str_representation(right_value, left_value, parameters.word_vectors).tolist()
        elif re.match(r'^__NOTEQUAL__', right_value) is not None:
            operator_idx = parameters.compare_ops_id['!=']
            operator_vec = [0 for _ in range(parameters.compare_ops_total_num)]
            operator_vec[operator_idx - 1] = 1
            right_value = right_value.strip('\'')[12:]
            right_value_vec = get_str_representation(right_value, left_value, parameters.word_vectors).tolist()
        elif re.match(r'^__ANY__', right_value) is not None:
            operator_idx = parameters.compare_ops_id['=']
            operator_vec = [0 for _ in range(parameters.compare_ops_total_num)]
            operator_vec[operator_idx - 1] = 1
            right_value = right_value.strip('\'')[7:].strip('{}')
            right_value_vec = []
            count = 0
            for v in right_value.split(','):
                v = v.strip('"').strip('\'')
                if len(v) > 0:
                    count += 1
                    vec = get_str_representation(v, left_value, parameters.word_vectors).tolist()
                    if len(right_value_vec) == 0:
                        right_value_vec = [0 for _ in vec]
                    for idx, vv in enumerate(vec):
                        right_value_vec[idx] += vv
            for idx in range(len(right_value_vec)):
                right_value_vec[idx] /= len(right_value.split(','))
        elif right_value == 'None':
            operator_idx = parameters.compare_ops_id['!Null']
            operator_vec = [0 for _ in range(parameters.compare_ops_total_num)]
            operator_vec[operator_idx - 1] = 1
            if operator == 'IS':
                right_value_vec = [1]
            elif operator == '!=':
                right_value_vec = [0]
            else:
                print(operator)
                raise
        else:
            #             print (left_value, operator, right_value)
            operator_idx = parameters.compare_ops_id[operator]
            operator_vec = [0 for _ in range(parameters.compare_ops_total_num)]
            operator_vec[operator_idx - 1] = 1
            right_value_vec = get_str_representation(right_value, left_value, parameters.word_vectors).tolist()
        vec = [0 for _ in range(parameters.bool_ops_total_num)]
        vec = vec + left_value_vec + operator_vec + right_value_vec
    num_pad = parameters.condition_op_dim - len(vec)
    result = np.pad(vec, (0, num_pad), 'constant')
    #     print 'condition op: ', result
    return result


def encode_condition(condition, relation_name, index_name, parameters):
    if len(condition) == 0:
        vecs = [[0 for _ in range(parameters.condition_op_dim)]]
    else:
        vecs = [encode_condition_op(condition_op, relation_name, index_name, parameters) for condition_op in condition]
    num_pad = parameters.condition_max_num - len(vecs)
    result = np.pad(vecs, ((0, num_pad), (0, 0)), 'constant')
    return result

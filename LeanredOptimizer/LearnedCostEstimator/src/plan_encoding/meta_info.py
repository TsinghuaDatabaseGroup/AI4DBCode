from gensim.models import Word2Vec
from gensim.models import KeyedVectors

import json
import math

def load_dictionary(path):
    word_vectors = KeyedVectors.load(path, mmap='r')
    return word_vectors

def load_numeric_min_max(path):
    with open(path,'r') as f:
        min_max_column = json.loads(f.read())
    return min_max_column

def determine_prefix(column):
    relation_name = column.split('.')[0]
    column_name = column.split('.')[1]
    if relation_name == 'aka_title':
        if column_name == 'title':
            return 'title_'
        else:
            print (column)
            raise
    elif relation_name == 'char_name':
        if column_name == 'name':
            return 'name_'
        elif column_name == 'name_pcode_nf':
            return 'nf_'
        elif column_name == 'surname_pcode':
            return 'surname_'
        else:
            print (column)
            raise
    elif relation_name == 'movie_info_idx':
        if column_name == 'info':
            return 'info_'
        else:
            print (column)
            raise
    elif relation_name == 'title':
        if column_name == 'title':
            return 'title_'
        else:
            print (column)
            raise
    elif relation_name == 'role_type':
        if column_name == 'role':
            return 'role_'
        else:
            print (column)
            raise
    elif relation_name == 'movie_companies':
        if column_name == 'note':
            return 'note_'
        else:
            print (column)
            raise
    elif relation_name == 'info_type':
        if column_name == 'info':
            return 'info_'
        else:
            print (column)
            raise
    elif relation_name == 'company_type':
        if column_name == 'kind':
            return ''
        else:
            print (column)
            raise
    elif relation_name == 'company_name':
        if column_name == 'name':
            return 'cn_name_'
        elif column_name == 'country_code':
            return 'country_'
        else:
            print (column)
            raise
    elif relation_name == 'keyword':
        if column_name == 'keyword':
            return 'keyword_'
        else:
            print (column)
            raise

    elif relation_name == 'movie_info':
        if column_name == 'info':
            return ''
        elif column_name == 'note':
            return 'note_'
        else:
            print (column)
            raise
    elif relation_name == 'name':
        if column_name == 'gender':
            return 'gender_'
        elif column_name == 'name':
            return 'name_'
        elif column_name == 'name_pcode_cf':
            return 'cf_'
        elif column_name == 'name_pcode_nf':
            return 'nf_'
        elif column_name == 'surname_pcode':
            return 'surname_'
        else:
            print (column)
            raise
    elif relation_name == 'aka_name':
        if column_name == 'name':
            return 'name_'
        elif column_name == 'name_pcode_cf':
            return 'cf_'
        elif column_name == 'name_pcode_nf':
            return 'nf_'
        elif column_name == 'surname_pcode':
            return 'surname_'
        else:
            print (column)
            raise
    elif relation_name == 'link_type':
        if column_name == 'link':
            return 'link_'
        else:
            print (column)
            raise
    elif relation_name == 'person_info':
        if column_name == 'note':
            return 'note_'
        else:
            print (column)
            raise
    elif relation_name == 'cast_info':
        if column_name == 'note':
            return 'note_'
        else:
            print (column)
            raise
    elif relation_name == 'comp_cast_type':
        if column_name == 'kind':
            return 'kind_'
        else:
            print (column)
            raise
    elif relation_name == 'kind_type':
        if column_name == 'kind':
            return 'kind_'
        else:
            print (column)
            raise
    else:
        print (column)
        raise

def obtain_upper_bound_query_size(path):
    plan_node_max_num = 0
    condition_max_num = 0
    cost_label_max = 0.0
    cost_label_min = 9999999999.0
    card_label_max = 0.0
    card_label_min = 9999999999.0
    plans = []
    with open(path, 'r') as f:
        for plan in f.readlines():
            plan = json.loads(plan)
            plans.append(plan)
            cost = plan['cost']
            cardinality = plan['cardinality']
            if cost > cost_label_max:
                cost_label_max = cost
            elif cost < cost_label_min:
                cost_label_min = cost
            if cardinality > card_label_max:
                card_label_max = cardinality
            elif cardinality < card_label_min:
                card_label_min = cardinality
            sequence = plan['seq']
            plan_node_num = len(sequence)
            if plan_node_num > plan_node_max_num:
                plan_node_max_num = plan_node_num
            for node in sequence:
                if node == None:
                    continue
                if 'condition_filter' in node:
                    condition_num = len(node['condition_filter'])
                    if condition_num > condition_max_num:
                        condition_max_num = condition_num
                if 'condition_index' in node:
                    condition_num = len(node['condition_index'])
                    if condition_num > condition_max_num:
                        condition_max_num = condition_num
    cost_label_min, cost_label_max = math.log(cost_label_min), math.log(cost_label_max)
    card_label_min, card_label_max = math.log(card_label_min), math.log(card_label_max)
    print (plan_node_max_num, condition_max_num)
    print (cost_label_min, cost_label_max)
    print (card_label_min, card_label_max)
    return plan_node_max_num, condition_max_num, cost_label_min, cost_label_max, card_label_min, card_label_max

def prepare_dataset(database):

    column2pos = dict()

    tables = ['aka_name', 'aka_title', 'cast_info', 'char_name', 'company_name', 'company_type', 'comp_cast_type', 'complete_cast', 'info_type', 'keyword', 'kind_type', 'link_type', 'movie_companies', 'movie_info', 'movie_info_idx',
              'movie_keyword', 'movie_link', 'name', 'person_info', 'role_type', 'title']

    for table_name in tables:
        column2pos[table_name] = database[table_name].columns

    indexes = ['aka_name_pkey', 'aka_title_pkey', 'cast_info_pkey', 'char_name_pkey',
               'comp_cast_type_pkey', 'company_name_pkey', 'company_type_pkey', 'complete_cast_pkey',
               'info_type_pkey', 'keyword_pkey', 'kind_type_pkey', 'link_type_pkey', 'movie_companies_pkey',
               'movie_info_idx_pkey', 'movie_keyword_pkey', 'movie_link_pkey', 'name_pkey', 'role_type_pkey',
               'title_pkey', 'movie_info_pkey', 'person_info_pkey', 'company_id_movie_companies',
               'company_type_id_movie_companies', 'info_type_id_movie_info_idx', 'info_type_id_movie_info',
               'info_type_id_person_info', 'keyword_id_movie_keyword', 'kind_id_aka_title', 'kind_id_title',
               'linked_movie_id_movie_link', 'link_type_id_movie_link', 'movie_id_aka_title', 'movie_id_cast_info',
               'movie_id_complete_cast', 'movie_id_movie_ companies', 'movie_id_movie_info_idx',
               'movie_id_movie_keyword', 'movie_id_movie_link', 'movie_id_movie_info', 'person_id_aka_name',
               'person_id_cast_info', 'person_id_person_info', 'person_role_id_cast_info', 'role_id_cast_info']
    indexes_id = dict()
    for idx, index in enumerate(indexes):
        indexes_id[index] = idx + 1
    physic_ops_id = {'Materialize':1, 'Sort':2, 'Hash':3, 'Merge Join':4, 'Bitmap Index Scan':5,
                     'Index Only Scan':6, 'BitmapAnd':7, 'Nested Loop':8, 'Aggregate':9, 'Result':10,
                     'Hash Join':11, 'Seq Scan':12, 'Bitmap Heap Scan':13, 'Index Scan':14, 'BitmapOr':15}
    strategy_id = {'Plain':1}
    compare_ops_id = {'=':1, '>':2, '<':3, '!=':4, '~~':5, '!~~':6, '!Null': 7, '>=':8, '<=':9}
    bool_ops_id = {'AND':1,'OR':2}
    tables_id = {}
    columns_id = {}
    table_id = 1
    column_id = 1
    for table_name in tables:
        tables_id[table_name] = table_id
        table_id += 1
        for column in column2pos[table_name]:
            columns_id[table_name+'.'+column] = column_id
            column_id += 1
    return column2pos, indexes_id, tables_id, columns_id, physic_ops_id, compare_ops_id, bool_ops_id, tables
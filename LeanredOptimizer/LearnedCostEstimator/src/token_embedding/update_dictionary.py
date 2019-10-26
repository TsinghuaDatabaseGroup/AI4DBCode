import json
import re

import pandas as pd

data = {}
data["aka_name"] = pd.read_csv('/home/sunji/imdb_data_csv/aka_name.csv', header=None)
data["aka_title"] = pd.read_csv('/home/sunji/imdb_data_csv/aka_title.csv', header=None)
data["cast_info"] = pd.read_csv('/home/sunji/imdb_data_csv/cast_info.csv', header=None)
data["char_name"] = pd.read_csv('/home/sunji/imdb_data_csv/char_name.csv', header=None)
data["company_name"] = pd.read_csv('/home/sunji/imdb_data_csv/company_name.csv', header=None)
data["company_type"] = pd.read_csv('/home/sunji/imdb_data_csv/company_type.csv', header=None)
data["comp_cast_type"] = pd.read_csv('/home/sunji/imdb_data_csv/comp_cast_type.csv', header=None)
data["complete_cast"] = pd.read_csv('/home/sunji/imdb_data_csv/complete_cast.csv', header=None)
data["info_type"] = pd.read_csv('/home/sunji/imdb_data_csv/info_type.csv', header=None)
data["keyword"] = pd.read_csv('/home/sunji/imdb_data_csv/keyword.csv', header=None)
data["kind_type"] = pd.read_csv('/home/sunji/imdb_data_csv/kind_type.csv', header=None)
data["link_type"] = pd.read_csv('/home/sunji/imdb_data_csv/link_type.csv', header=None)
data["movie_companies"] = pd.read_csv('/home/sunji/imdb_data_csv/movie_companies.csv', header=None)
data["movie_info"] = pd.read_csv('/home/sunji/imdb_data_csv/movie_info.csv', header=None)
data["movie_info_idx"] = pd.read_csv('/home/sunji/imdb_data_csv/movie_info_idx.csv', header=None)
data["movie_keyword"] = pd.read_csv('/home/sunji/imdb_data_csv/movie_keyword.csv', header=None)
data["movie_link"] = pd.read_csv('/home/sunji/imdb_data_csv/movie_link.csv', header=None)
data["name"] = pd.read_csv('/home/sunji/imdb_data_csv/name.csv', header=None)
data["person_info"] = pd.read_csv('/home/sunji/imdb_data_csv/person_info.csv', header=None)
data["role_type"] = pd.read_csv('/home/sunji/imdb_data_csv/role_type.csv', header=None)
data["title"] = pd.read_csv('/home/sunji/imdb_data_csv/title.csv', header=None)

aka_name_column = [
    'id',
    'person_id',
    'name',
    'imdb_index',
    'name_pcode_cf',
    'name_pcode_nf',
    'surname_pcode',
    'md5sum'
]
aka_title_column = [
    'id',
    'movie_id',
    'title',
    'imdb_index',
    'kind_id',
    'production_year',
    'phonetic_code',
    'episode_of_id',
    'season_nr',
    'episode_nr',
    'note',
    'md5sum'
]
cast_info_column = [
    'id',
    'person_id',
    'movie_id',
    'person_role_id',
    'note',
    'nr_order',
    'role_id'
]
char_name_column = [
    'id',
    'name',
    'imdb_index',
    'imdb_id',
    'name_pcode_nf',
    'surname_pcode',
    'md5sum'
]
comp_cast_type_column = [
    'id',
    'kind'
]
company_name_column = [
    'id',
    'name',
    'country_code',
    'imdb_id',
    'name_pcode_nf',
    'name_pcode_sf',
    'md5sum'
]
company_type_column = [
    'id',
    'kind'
]
complete_cast_column = [
    'id',
    'movie_id',
    'subject_id',
    'status_id'
]
info_type_column = [
    'id',
    'info'
]
keyword_column = [
    'id',
    'keyword',
    'phonetic_code'
]
kind_type_column = [
    'id',
    'kind'
]
link_type_column = [
    'id',
    'link'
]
movie_companies_column = [
    'id',
    'movie_id',
    'company_id',
    'company_type_id',
    'note'
]
movie_info_idx_column = [
    'id',
    'movie_id',
    'info_type_id',
    'info',
    'note'
]
movie_keyword_column = [
    'id',
    'movie_id',
    'keyword_id'
]
movie_link_column = [
    'id',
    'movie_id',
    'linked_movie_id',
    'link_type_id'
]
name_column = [
    'id',
    'name',
    'imdb_index',
    'imdb_id',
    'gender',
    'name_pcode_cf',
    'name_pcode_nf',
    'surname_pcode',
    'md5sum'
]
role_type_column = [
    'id',
    'role'
]
title_column = [
    'id',
    'title',
    'imdb_index',
    'kind_id',
    'production_year',
    'imdb_id',
    'phonetic_code',
    'episode_of_id',
    'season_nr',
    'episode_nr',
    'series_years',
    'md5sum'
]
movie_info_column = [
    'id',
    'movie_id',
    'info_type_id',
    'info',
    'note'
]
person_info_column = [
    'id',
    'person_id',
    'info_type_id',
    'info',
    'note'
]
data["aka_name"].columns = aka_name_column
data["aka_title"].columns = aka_title_column
data["cast_info"].columns = cast_info_column
data["char_name"].columns = char_name_column
data["company_name"].columns = company_name_column
data["company_type"].columns = company_type_column
data["comp_cast_type"].columns = comp_cast_type_column
data["complete_cast"].columns = complete_cast_column
data["info_type"].columns = info_type_column
data["keyword"].columns = keyword_column
data["kind_type"].columns = kind_type_column
data["link_type"].columns = link_type_column
data["movie_companies"].columns = movie_companies_column
data["movie_info"].columns = movie_info_column
data["movie_info_idx"].columns = movie_info_idx_column
data["movie_keyword"].columns = movie_keyword_column
data["movie_link"].columns = movie_link_column
data["name"].columns = name_column
data["person_info"].columns = person_info_column
data["role_type"].columns = role_type_column
data["title"].columns = title_column

train_like = {}
with open('/home/sunji/learnedcardinality/job/job_train_plan_seq_sample_big_150k_noagg.json') as f:
    for plan in f.readlines():
        plan = json.loads(plan)
        for node in plan['seq']:
            if node != None:
                if 'condition' in node:
                    for predicate in node['condition']:
                        if predicate != None and predicate['op_type'] == 'Compare' and (
                            predicate['right_value'].startswith('__LIKE__') or predicate['right_value'].startswith(
                                '__NOTLIKE__')):
                            relation_name = predicate['left_value'].split('.')[0]
                            column_name = predicate['left_value'].split('.')[1]
                            right_value = predicate['right_value']
                            if re.match(r'^__LIKE__', right_value):
                                right_value = right_value[8:]
                            elif re.match(r'^__NOTLIKE__', right_value):
                                right_value = right_value[11:]
                            else:
                                raise
                            if relation_name in train_like:
                                train_like[relation_name].add((column_name, right_value))
                            else:
                                train_like[relation_name] = set([(column_name, right_value)])
                if 'condition_filter' in node:
                    for predicate in node['condition_filter']:
                        if predicate != None and predicate['op_type'] == 'Compare' and (
                            predicate['right_value'].startswith('__LIKE__') or predicate['right_value'].startswith(
                                '__NOTLIKE__')):
                            relation_name = predicate['left_value'].split('.')[0]
                            column_name = predicate['left_value'].split('.')[1]
                            right_value = predicate['right_value']
                            if re.match(r'^__LIKE__', right_value):
                                right_value = right_value[8:]
                            elif re.match(r'^__NOTLIKE__', right_value):
                                right_value = right_value[11:]
                            else:
                                raise
                            if relation_name in train_like:
                                train_like[relation_name].add((column_name, right_value))
                            else:
                                train_like[relation_name] = set([(column_name, right_value)])
                if 'condition_index' in node:
                    for predicate in node['condition_index']:
                        if predicate != None and predicate['op_type'] == 'Compare' and (
                            predicate['right_value'].startswith('__LIKE__') or predicate['right_value'].startswith(
                                '__NOTLIKE__')):
                            relation_name = predicate['left_value'].split('.')[0]
                            column_name = predicate['left_value'].split('.')[1]
                            right_value = predicate['right_value']
                            if re.match(r'^__LIKE__', right_value):
                                right_value = right_value[8:]
                            elif re.match(r'^__NOTLIKE__', right_value):
                                right_value = right_value[11:]
                            else:
                                raise
                            if relation_name in train_like:
                                train_like[relation_name].add((column_name, right_value))
                            else:
                                train_like[relation_name] = set([(column_name, right_value)])

sentences = []
for idx, row in enumerate(data['name'].itertuples(), 1):
    #     print (ddd)
    if idx % 100 == 0:
        print('name', idx, '/', len(data['name']))
    sentence = set([])
    sentence.add('n_id_' + str(row.id))
    xxx = str(row.name)
    yyy = str(row.name_pcode_cf)
    for column_name, v in train_like['name']:
        if column_name == 'name':
            ddd = xxx
        else:
            ddd = yyy
        passed = True
        for token in v.split('%'):
            if len(token) > 0:
                if not token in ddd:
                    passed = False
                    break
        if passed:
            for token in v.split('%'):
                if len(token) > 0:
                    if column_name == 'name_pcode_cf':
                        token = 'cf_' + token
                        if not token in sentence:
                            sentence.add(token)
                    else:
                        token = 'name_' + token
                        if not token in sentence:
                            sentence.add(token)
    if len(sentence) > 1:
        sentences.append(list(sentence))

for idx, row in data['aka_name'].iterrows():
    if idx % 100 == 0:
        print('aka_name', idx, '/', len(data['aka_name']))
    sentence = set([])
    sentence.add('an_id_' + str(row['id']))
    ddd = str(row['name'])
    for column_name, v in train_like['aka_name']:
        passed = True
        for token in v.split('%'):
            if len(token) > 0:
                if not token in ddd:
                    passed = False
                    break
        if passed:
            for token in v.split('%'):
                if len(token) > 0:
                    token = 'name_' + token
                    if not token in sentence:
                        sentence.add(token)
    if len(sentence) > 1:
        sentences.append(list(sentence))

for idx, row in data['movie_companies'].iterrows():
    if idx % 100 == 0:
        print('movie_companies', idx, '/', len(data['movie_companies']))
    sentence = set([])
    sentence.add('mc_id_' + str(row['id']))
    sentence.add('m_id_' + str(row['movie_id']))
    sentence.add('c_id_' + str(row['company_id']))
    sentence.add('ct_id_' + str(row['company_type_id']))
    ddd = str(row['note'])
    for column_name, v in train_like['movie_companies']:
        passed = True
        for token in v.split('%'):
            if len(token) > 0:
                if not token in ddd:
                    passed = False
                    break
        if passed:
            for token in v.split('%'):
                if len(token) > 0:
                    token = 'note_' + token
                    if not token in sentence:
                        sentence.add(token)
    if len(sentence) > 4:
        sentences.append(list(sentence))

for idx, row in data['cast_info'].iterrows():
    if idx % 100 == 0:
        print('cast_info', idx, '/', len(data['cast_info']))
    sentence = set([])
    sentence.add('ci_id_' + str(row['id']))
    sentence.add('p_id_' + str(row['person_id']))
    sentence.add('m_id_' + str(row['movie_id']))
    sentence.add('pr_id_' + str(row['person_role_id']))
    sentence.add('r_id_' + str(row['role_id']))
    ddd = str(row[column_name])
    for column_name, v in train_like['cast_info']:
        passed = True
        for token in v.split('%'):
            if len(token) > 0:
                if not token in ddd:
                    passed = False
                    break
        if passed:
            for token in v.split('%'):
                if len(token) > 0:
                    token = 'note_' + token
                    if not token in sentence:
                        sentence.add(token)
    if len(sentence) > 5:
        sentences.append(list(sentence))

for idx, row in data['keyword'].iterrows():
    if idx % 100 == 0:
        print('keyword', idx, '/', len(data['keyword']))
    sentence = set([])
    sentence.add('key_id_' + str(row['id']))
    ddd = str(row['keyword'])
    for column_name, v in train_like['keyword']:
        passed = True
        for token in v.split('%'):
            if len(token) > 0:
                if not token in ddd:
                    passed = False
                    break
        if passed:
            for token in v.split('%'):
                if len(token) > 0:
                    token = 'keyword_' + token
                    if not token in sentence:
                        sentence.add(token)
    if len(sentence) > 1:
        sentences.append(list(sentence))

for idx, row in data['title'].iterrows():
    if idx % 100 == 0:
        print('title', idx, '/', len(data['title']))
    sentence = set([])
    sentence.add('t_id_' + str(row['id']))
    sentence.add('k_id_' + str(row['kind_id']))
    ddd = str(row['title'])
    for column_name, v in train_like['title']:
        passed = True
        for token in v.split('%'):
            if len(token) > 0:
                if not token in ddd:
                    passed = False
                    break
        if passed:
            for token in v.split('%'):
                if len(token) > 0:
                    token = 'title_' + token
                    if not token in sentence:
                        sentence.add(token)
    if len(sentence) > 2:
        sentences.append(list(sentence))

for idx, row in data['char_name'].iterrows():
    if idx % 100 == 0:
        print('char_name', idx, '/', len(data['char_name']))
    sentence = set([])
    sentence.add('chn_id_' + str(row['id']))
    ddd = str(row['name'])
    for column_name, v in train_like['char_name']:
        passed = True
        for token in v.split('%'):
            if len(token) > 0:
                if not token in ddd:
                    passed = False
                    break
        if passed:
            for token in v.split('%'):
                if len(token) > 0:
                    token = 'name_' + token
                    if not token in sentence:
                        sentence.add(token)
    if len(sentence) > 1:
        sentences.append(list(sentence))

for idx, row in data['company_name'].iterrows():
    if idx % 100 == 0:
        print('company_name', idx, '/', len(data['company_name']))
    sentence = set([])
    sentence.add('cn_id_' + str(row['id']))
    ddd = str(row['name'])
    for column_name, v in train_like['company_name']:
        passed = True
        for token in v.split('%'):
            if len(token) > 0:
                if not token in ddd:
                    passed = False
                    break
        if passed:
            for token in v.split('%'):
                if len(token) > 0:
                    token = 'cn_name_' + token
                    if not token in sentence:
                        sentence.add(token)
    if len(sentence) > 1:
        sentences.append(list(sentence))

for idx, row in data['movie_info'].iterrows():
    if idx % 100 == 0:
        print('movie_info', idx, '/', len(data['movie_info']))
    sentence = set([])
    sentence.add('mi_id_' + str(row['id']))
    sentence.add('m_id_' + str(row['movie_id']))
    sentence.add('it_id_' + str(row['info_type_id']))
    xxx = str(row['info'])
    yyy = str(row['note'])
    for column_name, v in train_like['movie_info']:
        if column_name == 'info':
            ddd = xxx
            prefix = ''
        else:
            ddd = yyy
            prefix = 'note_'
        passed = True
        for token in v.split('%'):
            if len(token) > 0:
                if not token in ddd:
                    passed = False
                    break
        if passed:
            for token in v.split('%'):
                if len(token) > 0:
                    token = prefix + token
                    if not token in sentence:
                        sentence.add(token)
    if len(sentence) > 3:
        sentences.append(list(sentence))

for idx, row in data['link_type'].iterrows():
    if idx % 100 == 0:
        print('link_type', idx, '/', len(data['link_type']))
    sentence = set([])
    sentence.add('it_id_' + str(row['id']))
    ddd = str(row['link'])
    for column_name, v in train_like['link_type']:
        passed = True
        for token in v.split('%'):
            if len(token) > 0:
                if not token in ddd:
                    passed = False
                    break
        if passed:
            for token in v.split('%'):
                if len(token) > 0:
                    token = 'link_' + token
                    if not token in sentence:
                        sentence.add(token)
    if len(sentence) > 1:
        sentences.append(list(sentence))

print(len(sentences))
import pickle

pickle.dump(sentences, open('/home/sunji/learnedcardinality/string_words/more_sentences_train_query_big.pkl', 'wb'))

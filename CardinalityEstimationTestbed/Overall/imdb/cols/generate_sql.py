import random

import pandas as pd
from numpy.random import choice

dict1 = {'title': ['t', 'id', 'imdb_index', 'kind_id', 'production_year', 'phonetic_code', 'season_nr', 'episode_nr',
                   'series_years'],
         'movie_info_idx': ['mi_idx', 'movie_id', 'info_type_id'],
         'movie_info': ['mi', 'movie_id', 'info_type_id'],
         'cast_info': ['ci', 'movie_id', 'nr_order', 'role_id'],
         'movie_keyword': ['mk', 'movie_id', 'keyword_id'],
         'movie_companies': ['mc', 'movie_id', 'company_type_id']}

dict2 = {'title': ['t', 'id', 'kind_id', 'production_year', 'phonetic_code', 'season_nr', 'episode_nr'],
         'movie_info_idx': ['mi_idx', 'movie_id', 'info_type_id'],
         'movie_info': ['mi', 'movie_id', 'info_type_id'],
         'cast_info': ['ci', 'movie_id', 'nr_order', 'role_id'],
         'movie_keyword': ['mk', 'movie_id', 'keyword_id'],
         'movie_companies': ['mc', 'movie_id', 'company_type_id']}  # 

df_title = pd.read_csv('.../train-test-data/imdbdata-num/title.csv', sep=',', escapechar='\\', encoding='utf-8',
                       low_memory=False, quotechar='"',
                       usecols=['id', 'kind_id', 'production_year', 'phonetic_code', 'season_nr', 'episode_nr'])
df_cast_info = pd.read_csv('.../train-test-data/imdbdata-num/cast_info.csv', sep=',', escapechar='\\', encoding='utf-8',
                           low_memory=False, quotechar='"', error_bad_lines=False,
                           usecols=['movie_id', 'nr_order', 'role_id'])
df_movie_companies = pd.read_csv('.../train-test-data/imdbdata-num/movie_companies.csv', sep=',', escapechar='\\',
                                 encoding='utf-8', low_memory=False, quotechar='"',
                                 usecols=['movie_id', 'company_type_id'])
df_movie_info = pd.read_csv('.../train-test-data/imdbdata-num/movie_info.csv', sep=',', escapechar='\\',
                            encoding='utf-8', low_memory=False, quotechar='"',
                            usecols=['movie_id', 'info_type_id'])
df_movie_info_idx = pd.read_csv('.../train-test-data/imdbdata-num/movie_info_idx.csv', sep=',', escapechar='\\',
                                encoding='utf-8', low_memory=False, quotechar='"',
                                usecols=['movie_id', 'info_type_id'])
df_movie_keyword = pd.read_csv('.../train-test-data/imdbdata-num/movie_keyword.csv', sep=',', escapechar='\\',
                               encoding='utf-8', low_memory=False, quotechar='"',
                               usecols=['movie_id', 'keyword_id'])

df_title = df_title.dropna(axis=0, how='any', inplace=False)
df_cast_info = df_cast_info.dropna(axis=0, how='any', inplace=False)
df_movie_companies = df_movie_companies.dropna(axis=0, how='any', inplace=False)
df_movie_info = df_movie_info.dropna(axis=0, how='any', inplace=False)
df_movie_info_idx = df_movie_info_idx.dropna(axis=0, how='any', inplace=False)
df_movie_keyword = df_movie_keyword.dropna(axis=0, how='any', inplace=False)

df_title_l = df_title.sample(frac=0.003, replace=False, random_state=1)
df_cast_info_l = df_cast_info.sample(frac=0.0002, replace=False, random_state=1)
df_movie_companies_l = df_movie_companies.sample(frac=0.0006, replace=False, random_state=1)
df_movie_info_l = df_movie_info.sample(frac=0.0002, replace=False, random_state=1)
df_movie_info_idx_l = df_movie_info_idx.sample(frac=0.0009, replace=False, random_state=1)
df_movie_keyword_l = df_movie_keyword.sample(frac=0.0005, replace=False, random_state=1)
t_imdb_index = [2, 3, 4, 5]
t_kind_id = []
t_production_year = []
t_phonetic_code = []
t_season_nr = []
t_episode_nr = []
t_series_years = []
mi_idx_info_type_id = []
mi_info_type_id = []
ci_nr_order = []
ci_role_id = []
mk_keyword_id = []
mc_company_type_id = []

for key, value in dict2.items():
    for i in range(2, len(value)):
        locals()[value[0] + '_' + value[i]] = list(locals()['df_' + key + '_l'][value[i]])

# Cols-4 production_year & phonetic_code & series_year & role_id

f2 = open("../train-test-data/imdb-cols-sql/4/4-all-str.sql", 'w')
# tnum = [2,3]  # 
tables = ['cast_info']  # 
ops = ['=', '<', '>']  # >=, <=

dictcols = {'title': ['production_year', 'phonetic_code', 'series_years'],
            'cast_info': ['role_id']}  #

dictalias = {'title': ['t'],
             'movie_info_idx': ['mi_idx'],
             'movie_info': ['mi'],
             'cast_info': ['ci'],
             'movie_keyword': ['mk'],
             'movie_companies': ['mc']}

dictjk = {'title': ['id'],
          'movie_info_idx': ['movie_id'],
          'movie_info': ['movie_id'],
          'cast_info': ['movie_id'],
          'movie_keyword': ['movie_id'],
          'movie_companies': ['movie_id']}

for i in range(40000):
    questr = 'SELECT COUNT(*) FROM '
    joinks = []
    tablenames = []  # title t 
    predicates = []

    num_tcol = random.randint(1, len(dictcols['title']))
    t1 = list(choice(dictcols['title'], num_tcol, replace=False))
    for k in range(num_tcol):
        t2 = locals()['t_' + t1[k]]
        predicates.append('t.' + str(t1[k]) + choice(ops) + str(int(choice(t2))))

    num_t = random.randint(0, len(tables))  # 
    tables1 = list(choice(tables, num_t, replace=False))
    for j in range(num_t):
        table1 = tables1[j]  # 
        joinks.append('t.id=' + dictalias[table1][0] + '.movie_id')
        tablenames.append(table1 + ' ' + dictalias[table1][0])

        num_tcol = random.randint(1, len(dictcols[table1]))
        t1 = list(choice(dictcols[table1], num_tcol, replace=False))
        for k in range(num_tcol):
            t2 = locals()[dictalias[table1][0] + '_' + t1[k]]
            predicates.append(dictalias[table1][0] + '.' + str(t1[k]) + choice(ops) + str(int(choice(t2))))

    for tn in tablenames:
        questr += tn + ',' + ' '
    questr += 'title t WHERE '

    for jks in joinks:
        questr += jks + ' AND '

    for pre in predicates:
        questr += pre + ' AND '
    questr = questr[:len(questr) - 5]
    questr += ';\n'

    f2.write(questr)
f2.close()

# Cols-6: production_year & phonetic_code & series_years & kind_id & role_id & info_type_id

f2 = open("../train-test-data/imdb-cols-sql/6/6-all-num.sql", 'w')

tables = ['movie_info', 'cast_info']  #
ops = ['=', '<', '>']  # >=, <=

dictcols = {'title': ['kind_id', 'production_year', 'phonetic_code', 'series_years'],
            'movie_info': ['info_type_id'],
            'cast_info': ['role_id']}  #

dictalias = {'title': ['t'],
             'movie_info_idx': ['mi_idx'],
             'movie_info': ['mi'],
             'cast_info': ['ci'],
             'movie_keyword': ['mk'],
             'movie_companies': ['mc']}

dictjk = {'title': ['id'],
          'movie_info_idx': ['movie_id'],
          'movie_info': ['movie_id'],
          'cast_info': ['movie_id'],
          'movie_keyword': ['movie_id'],
          'movie_companies': ['movie_id']}

for i in range(50000):
    questr = 'SELECT COUNT(*) FROM '
    joinks = []
    tablenames = []  # title t 
    predicates = []

    num_tcol = random.randint(1, len(dictcols['title']))
    t1 = list(choice(dictcols['title'], num_tcol, replace=False))
    for k in range(num_tcol):
        t2 = locals()['t_' + t1[k]]
        predicates.append('t.' + str(t1[k]) + choice(ops) + str(int(choice(t2))))

    num_t = random.randint(0, len(tables))
    tables1 = list(choice(tables, num_t, replace=False))
    for j in range(num_t):
        table1 = tables1[j]  # 
        joinks.append('t.id=' + dictalias[table1][0] + '.movie_id')
        tablenames.append(table1 + ' ' + dictalias[table1][0])

        num_tcol = random.randint(1, len(dictcols[table1]))
        t1 = list(choice(dictcols[table1], num_tcol, replace=False))
        for k in range(num_tcol):
            t2 = locals()[dictalias[table1][0] + '_' + t1[k]]
            predicates.append(dictalias[table1][0] + '.' + str(t1[k]) + choice(ops) + str(int(choice(t2))))

    for tn in tablenames:
        questr += tn + ',' + ' '
    questr += 'title t WHERE '

    for jks in joinks:
        questr += jks + ' AND '

    for pre in predicates:
        questr += pre + ' AND '
    questr = questr[:len(questr) - 5]
    questr += ';\n'

    f2.write(questr)
f2.close()

# Cols-8: production_year & phonetic_code & series_years & kind_id & role_id & info_type_id & nr_order & episode_nr

f2 = open("../train-test-data/imdb-cols-sql/8/8-all-num.sql", 'w')

tables = ['movie_info', 'cast_info']  #
ops = ['=', '<', '>']  # >=, <=

dictcols = {'title': ['kind_id', 'production_year', 'phonetic_code', 'episode_nr', 'series_years'],
            'movie_info': ['info_type_id'],
            'cast_info': ['nr_order', 'role_id']}  #

dictalias = {'title': ['t'],
             'movie_info_idx': ['mi_idx'],
             'movie_info': ['mi'],
             'cast_info': ['ci'],
             'movie_keyword': ['mk'],
             'movie_companies': ['mc']}

dictjk = {'title': ['id'],
          'movie_info_idx': ['movie_id'],
          'movie_info': ['movie_id'],
          'cast_info': ['movie_id'],
          'movie_keyword': ['movie_id'],
          'movie_companies': ['movie_id']}

for i in range(60000):
    questr = 'SELECT COUNT(*) FROM '
    joinks = []
    tablenames = []  # title t 
    predicates = []

    num_tcol = random.randint(1, len(dictcols['title']))
    t1 = list(choice(dictcols['title'], num_tcol, replace=False))
    for k in range(num_tcol):
        t2 = locals()['t_' + t1[k]]
        predicates.append('t.' + str(t1[k]) + choice(ops) + str(int(choice(t2))))

    num_t = random.randint(0, len(tables))  #
    tables1 = list(choice(tables, num_t, replace=False))
    for j in range(num_t):
        table1 = tables1[j]  # 
        joinks.append('t.id=' + dictalias[table1][0] + '.movie_id')
        tablenames.append(table1 + ' ' + dictalias[table1][0])

        num_tcol = random.randint(1, len(dictcols[table1]))
        t1 = list(choice(dictcols[table1], num_tcol, replace=False))
        for k in range(num_tcol):
            t2 = locals()[dictalias[table1][0] + '_' + t1[k]]
            predicates.append(dictalias[table1][0] + '.' + str(t1[k]) + choice(ops) + str(int(choice(t2))))

    for tn in tablenames:
        questr += tn + ',' + ' '
    questr += 'title t WHERE '

    for jks in joinks:
        questr += jks + ' AND '

    for pre in predicates:
        questr += pre + ' AND '
    questr = questr[:len(questr) - 5]
    questr += ';\n'

    f2.write(questr)
f2.close()

import pandas as pd
from numpy.random import choice

# high production_year & phonetic_code & series_year & role_id
dict2 = {'title': ['t', 'id', 'production_year', 'phonetic_code', 'series_years'],
         'cast_info': ['ci', 'movie_id', 'role_id']}

df_title = pd.read_csv(".../train-test-data/imdb-distinct-data/high/title.csv", sep=',', escapechar='\\',
                       encoding='utf-8', low_memory=False, quotechar='"')
df_cast_info = pd.read_csv(".../train-test-data/imdb-distinct-data/high/cast_info.csv", sep=',', escapechar='\\',
                           encoding='utf-8', low_memory=False, quotechar='"', error_bad_lines=False)

for key, value in dict2.items():
    for i in range(2, len(value)):
        locals()[value[0] + '_' + value[i]] = list(locals()['df_' + key][value[i]].unique())  #
        locals()[value[0] + '_' + value[i]].sort()
        for j in locals()[value[0] + '_' + value[i]]:
            if pd.isnull(j):
                locals()[value[0] + '_' + value[i]].remove(j)

f2 = open(".../train-test-data/imdb-distinct-data/high/distincthigh.sql", 'w')

tables = ['cast_info']
ops = ['=', '<', '>']  #

dictcols = {'title': ['production_year', 'phonetic_code', 'series_years'],
            'cast_info': ['role_id']}

dictalias = {'title': ['t'],
             'cast_info': ['ci']}

dictjk = {'title': ['id'],
          'cast_info': ['movie_id']}

for i in range(36000):
    questr = 'SELECT COUNT(*) FROM '
    joinks = []
    tablenames = []  # title t I added it later
    predicates = []

    num_tcol = len(dictcols['title'])  # num_tcol = random.randint(1,len(dictcols['title']))
    t1 = list(choice(dictcols['title'], num_tcol, replace=False))
    for k in range(num_tcol):
        t2 = locals()['t_' + t1[k]]
        predicates.append('t.' + str(t1[k]) + choice(ops) + str(int(choice(t2))))

    num_t = 1  # Additional tables are required
    tables1 = list(choice(tables, num_t, replace=False))
    for j in range(num_t):
        table1 = tables1[j]  # For each table
        joinks.append('t.id=' + dictalias[table1][0] + '.movie_id')
        tablenames.append(table1 + ' ' + dictalias[table1][0])

        num_tcol = len(dictcols[table1])  # num_tcol = random.randint(1,len(dictcols[table1]))
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

# mid production_year & phonetic_code & series_year & role_id
dict2 = {'title': ['t', 'id', 'production_year', 'phonetic_code', 'series_years'],
         'cast_info': ['ci', 'movie_id', 'role_id']}

df_title = pd.read_csv(".../train-test-data/imdb-distinct-data/mid/title.csv", sep=',', escapechar='\\',
                       encoding='utf-8', low_memory=False, quotechar='"')
df_cast_info = pd.read_csv(".../train-test-data/imdb-distinct-data/mid/cast_info.csv", sep=',', escapechar='\\',
                           encoding='utf-8', low_memory=False, quotechar='"', error_bad_lines=False)

for key, value in dict2.items():
    for i in range(2, len(value)):
        locals()[value[0] + '_' + value[i]] = list(locals()['df_' + key][value[i]].unique())  #
        locals()[value[0] + '_' + value[i]].sort()
        for j in locals()[value[0] + '_' + value[i]]:
            if pd.isnull(j):
                locals()[value[0] + '_' + value[i]].remove(j)

f2 = open(".../train-test-data/imdb-distinct-data/mid/distinctmid.sql", 'w')

tables = ['cast_info']
ops = ['=', '<', '>']  # >=, <=

dictcols = {'title': ['production_year', 'phonetic_code', 'series_years'],
            'cast_info': ['role_id']}

dictalias = {'title': ['t'],
             'cast_info': ['ci']}

dictjk = {'title': ['id'],
          'cast_info': ['movie_id']}

for i in range(36000):
    questr = 'SELECT COUNT(*) FROM '
    joinks = []
    tablenames = []  # title t 
    predicates = []

    num_tcol = len(dictcols['title'])  # num_tcol = random.randint(1,len(dictcols['title']))
    t1 = list(choice(dictcols['title'], num_tcol, replace=False))
    for k in range(num_tcol):
        t2 = locals()['t_' + t1[k]]
        predicates.append('t.' + str(t1[k]) + choice(ops) + str(int(choice(t2))))

    num_t = 1  # 
    tables1 = list(choice(tables, num_t, replace=False))
    for j in range(num_t):
        table1 = tables1[j]  # 
        joinks.append('t.id=' + dictalias[table1][0] + '.movie_id')
        tablenames.append(table1 + ' ' + dictalias[table1][0])

        num_tcol = len(dictcols[table1])  # num_tcol = random.randint(1,len(dictcols[table1]))
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

# low production_year & phonetic_code & series_year & role_id
dict2 = {'title': ['t', 'id', 'production_year', 'phonetic_code', 'series_years'],
         'cast_info': ['ci', 'movie_id', 'role_id']}

df_title = pd.read_csv(".../train-test-data/imdb-distinct-data/low/title.csv", sep=',', escapechar='\\',
                       encoding='utf-8', low_memory=False, quotechar='"')
df_cast_info = pd.read_csv(".../train-test-data/imdb-distinct-data/low/cast_info.csv", sep=',', escapechar='\\',
                           encoding='utf-8', low_memory=False, quotechar='"', error_bad_lines=False)

for key, value in dict2.items():
    for i in range(2, len(value)):
        locals()[value[0] + '_' + value[i]] = list(locals()['df_' + key][value[i]].unique())  #
        locals()[value[0] + '_' + value[i]].sort()
        for j in locals()[value[0] + '_' + value[i]]:
            if pd.isnull(j):
                locals()[value[0] + '_' + value[i]].remove(j)

f2 = open(".../train-test-data/imdb-distinct-data/low/distinctlow.sql", 'w')

tables = ['cast_info']
ops = ['=', '<', '>']  # >=, <=

dictcols = {'title': ['production_year', 'phonetic_code', 'series_years'],
            'cast_info': ['role_id']}

dictalias = {'title': ['t'],
             'cast_info': ['ci']}

dictjk = {'title': ['id'],
          'cast_info': ['movie_id']}

for i in range(36000):
    questr = 'SELECT COUNT(*) FROM '
    joinks = []
    tablenames = []  # 
    predicates = []

    num_tcol = len(dictcols['title'])  # num_tcol = random.randint(1,len(dictcols['title']))
    t1 = list(choice(dictcols['title'], num_tcol, replace=False))
    for k in range(num_tcol):
        t2 = locals()['t_' + t1[k]]
        predicates.append('t.' + str(t1[k]) + choice(ops) + str(int(choice(t2))))

    num_t = 1  # 
    tables1 = list(choice(tables, num_t, replace=False))
    for j in range(num_t):
        table1 = tables1[j]  # 
        joinks.append('t.id=' + dictalias[table1][0] + '.movie_id')
        tablenames.append(table1 + ' ' + dictalias[table1][0])

        num_tcol = len(dictcols[table1])
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

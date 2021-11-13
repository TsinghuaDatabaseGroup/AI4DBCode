import argparse

import pandas as pd
import psycopg2
from physical_db import DBConnection, TrueCardinalityEstimator
from tqdm import tqdm

parser = argparse.ArgumentParser(description='get_truecard')

parser.add_argument('--version', type=str, help='datasets_dir', default='cols_4_distinct_1000_corr_5_skew_5')

args = parser.parse_args()
version = args.version

# db_connection = DBConnection(db='postgres',db_user='postgres',db_host="/var/run/postgresql")  # modify
db_connection = DBConnection(db='autocard', db_password="jintao2020", db_user='jintao', db_host="localhost")
fschema = open('./csvdata_sql/schema_' + version + '.sql')
schemasql = fschema.read()
dropsql = 'DROP TABLE ' + version + ';'

try:
    db_connection.submit_query(dropsql)  # Clear the table with the current name
except Exception as e:
    pass

try:
    db_connection.submit_query(schemasql)  # establish schema
except Exception as e:
    pass

# csvsql = r'\copy ' + version + ' from ./csvdata_sql/' + version + '.csv with csv header;'
# db_connection.submit_query(csvsql)
# os.system('psql -U postgres')
# os.system(csvsql)
df = pd.read_csv('./csvdata_sql/' + version + '.csv', sep=',', escapechar='\\', encoding='utf-8', low_memory=False,
                 quotechar='"')
columns = tuple(df.columns)
# connection = psycopg2.connect(user='postgres', host="/var/run/postgresql", database='postgres')
connection = psycopg2.connect(user='jintao', host="localhost", database='autocard', password="jintao2020")
cur = connection.cursor()
file = open('./csvdata_sql/' + version + '_nohead.csv', 'r')  # Read a file without a header
cur.copy_from(file, version, sep=',')
connection.commit()

true_estimator = TrueCardinalityEstimator(db_connection)
#
f = open('./csvdata_sql/' + version + '.sql')
queries = f.readlines()
i = 0
ftrain = open('./sql_truecard/' + version + 'train.sql', 'w')
ftest = open('./sql_truecard/' + version + 'test.sql', 'w')
for query in tqdm(queries):
    try:
        cardinality_true = true_estimator.true_cardinality(query)
        # print(cardinality_true)
        if cardinality_true == 0:
            continue
        query = query[0:len(query) - 1]
        if i < 10000:
            ftrain.write(query + ',')
            ftrain.write(str(cardinality_true))
            ftrain.write('\n')
        else:
            ftest.write(query + ',')
            ftest.write(str(cardinality_true))
            ftest.write('\n')
        i = i + 1
        if i >= 11000:
            break
    except Exception as e:
        # f2.write('Pass '+query+'\n')
        pass
    continue
ftrain.close()
ftest.close()

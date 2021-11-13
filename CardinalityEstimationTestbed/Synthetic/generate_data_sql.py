import argparse

import numpy as np
import pandas as pd
from numpy.random import choice
from scipy.stats import genpareto

parser = argparse.ArgumentParser(description='generatedata')

parser.add_argument('--distinct', type=int, help='datasets_dir', default=1000)
parser.add_argument('--cols', type=int, help='datasets_dir', default=4)
parser.add_argument('--corr', type=float, help='datasets_dir', default=5)
parser.add_argument('--skew', type=float, help='datasets_dir', default=5)

args = parser.parse_args()

distinct = args.distinct
cols = args.cols
corr = args.corr
skew = args.skew

version = 'cols_' + str(cols) + '_distinct_' + str(args.distinct) + '_corr_' + str(int(corr)) + '_skew_' + str(
    int(skew))

corr = args.corr / 10
skew = args.skew / 10  # create table

path = './csvdata_sql'
csv_path = path + '/' f"{version}.csv"
csv_path2 = path + '/' f"{version}_nohead.csv"  # for deepdb
seed = 2

df = pd.DataFrame()
for i in range(cols - 1):
    seed = seed + 1
    row_num = 100000  # 
    np.random.seed(seed)
    # The first column is generated according to skew
    skewfunc = genpareto.rvs(skew - 1, size=row_num - distinct)  # probability distribution function
    skewfunc = ((skewfunc - skewfunc.min()) / (skewfunc.max() - skewfunc.min()))  # normalization
    skewfunc = skewfunc * distinct  # Satisfies the unique value condition
    intskewfunc = skewfunc.astype(int)
    col0 = np.concatenate((np.arange(distinct), np.clip(intskewfunc, 0,
                                                        distinct - 1)))  # Ensure that each field value has at least one value
    colother = []
    for n in col0:
        if np.random.uniform(0, 1) <= corr:
            colother.append(n)
        else:
            colother.append(np.random.choice(distinct))  # corr
    if i == 0:
        df['col' + str(2 * i)] = col0
        df['col' + str(2 * i + 1)] = colother
    else:
        df['col' + str(i + 1)] = colother
df.to_csv(csv_path, index=False)
df.to_csv(csv_path2, index=False, header=False)  # using for deepdb

ops = ['=', '<', '>']  # train and test all not contain >=, <=
f2 = open('./csvdata_sql/' + version + '.sql', 'w')
for i in range(360000):  # as much as possible sqls
    sql = 'SELECT COUNT(*) FROM ' + version + ' cdcs WHERE '
    for i in range(cols):
        sql += 'cdcs.col' + str(i) + choice(ops) + str(list(np.random.randint(0, distinct, 1))[0]) + ' AND '
    sql = sql[0: len(sql) - 5]
    sql = sql + ';\n'
    f2.write(sql)
f2.close()

f3 = open('./csvdata_sql/schema_' + version + '.sql', 'w')
sql = 'CREATE TABLE ' + version + '(\n'
for i in range(cols):
    sql += '    col' + str(i) + ' integer NOT NULL,\n'
sql = sql[0: len(sql) - 2] + '\n);'
f3.write(sql)
f3.close()

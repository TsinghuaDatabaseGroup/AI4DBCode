import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

f2 = open('/home/zhangjintao/DeepDB/ranges/deepdb_job_ranges/data_preparation/chi2.txt', 'a')
df1 = pd.read_csv('/home/zhangjintao/DeepDB/ranges/imdb-benchmark/ranges_sample_0.15*2_0.03_join.csv', sep=',',
                  escapechar='\\', encoding='utf-8', low_memory=False, quotechar='"')
df1 = df1.sample(frac=0.06, replace=False, random_state=1)  # 0.15every  0.15*2  0.03*0.06
for i in df1.columns.values:
    for j in df1.columns.values:
        if (i == j):
            continue
        df2 = df1[[i, j]]
        df2 = df2.dropna(axis=0, how='any', inplace=False)
        if (len(df2[i]) == 0 or len(df2[j]) == 0):
            print('skip because len = 0')
            continue
        X = pd.get_dummies(df2[i])
        y = df2[j]
        # X=X.fillna(0)
        # y=y.fillna(0)
        sk = SelectKBest(chi2, k='all')
        sk.fit(X, y)
        str_score = str(np.sum(sk.scores_))
        len_x = len(df2[i].unique()) - 1
        len_y = len(df2[j].unique()) - 1
        str_len_x = str(len_x)
        str_len_y = str(len_y)
        crit = stats.chi2.ppf(q=0.99, df=(len_x - 1) * (len_y - 1))
        str_write = i + ' & ' + j + ' chi2 is: ' + str_score + '  lenx: ' + str_len_x + '  leny: ' + str_len_y + ' crit0.99: ' + str(
            crit) + '\n'
        # print(str_write)
        f2.write(str_write)
        print(i, '&', j, 'chi2 is:', np.sum(sk.scores_), '  lenx:', str_len_x, 'leny:', str_len_y, '  crit0.99:', crit)

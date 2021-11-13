import numpy as np
import pandas as pd
from tqdm import tqdm

df = pd.read_csv("../train-test-data/imdbdataset-str/title.csv", sep=',', escapechar='\\', encoding='utf-8',
                 low_memory=False)
res = dict()
df.head()

data_array = np.array(df['phonetic_code'])
data_list = data_array.tolist()
data_list = list(set(data_list))
if np.nan in data_list:
    data_list.remove(np.nan)
data_list = [str(i) for i in data_list]
data_list.sort()
i = 1
for key in data_list:
    res[key] = i
    i += 1
for i in tqdm(range(len(df['phonetic_code']))):
    if (pd.notnull(df['phonetic_code'][i])):
        df['phonetic_code'][i] = res[df['phonetic_code'][i]]

data_array = np.array(df['series_years'])
data_list = data_array.tolist()
data_list = list(set(data_list))
if np.nan in data_list:
    data_list.remove(np.nan)
data_list = [str(i) for i in data_list]
data_list.sort()
i = 1
for key in data_list:
    res[key] = i
    i += 1
for i in tqdm(range(len(df['series_years']))):
    if (pd.notnull(df['series_years'][i])):
        df['series_years'][i] = res[df['series_years'][i]]

data_array = np.array(df['imdb_index'])
data_list = data_array.tolist()
data_list = list(set(data_list))
if np.nan in data_list:
    data_list.remove(np.nan)
data_list = [str(i) for i in data_list]
data_list.sort()
i = 1
for key in data_list:
    res[key] = i
    i += 1
for i in tqdm(range(len(df['imdb_index']))):
    if (pd.notnull(df['imdb_index'][i])):
        df['imdb_index'][i] = res[df['imdb_index'][i]]

df.to_csv("../train-test-data/imdbdataset-num/title.csv", header=False, index=False)

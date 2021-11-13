import os
import pickle
import re

import pandas as pd

min_max_file = '/home/jintao/CardinalityEstimationBenchmark/learnedcardinalities-master/data/column_min_max_vals.csv'


def preprocess_sql(sql_path):
    output = []
    cols = set([])
    with open(sql_path, 'r') as f:
        for line in f.readlines():
            print(line)
            sql = ','.join(line.split(',')[:-1])
            cardinality = line.split(',')[-1].strip('\n')
            tables = [x.strip() for x in re.search('FROM(.*)WHERE', sql, re.IGNORECASE).group(1).split(',')]
            joins = [x.strip() for x in
                     re.search('WHERE(.*)', sql, re.IGNORECASE).group(1).split('AND')[0:len(tables) - 1]]
            conditions = [x.strip(' ;\n') for x in
                          re.search('WHERE(.*)', sql, re.IGNORECASE).group(1).split('AND')[len(tables) - 1:]]
            conds = []
            for cond in conditions:
                operator = re.search('([<>=])', cond, re.IGNORECASE).group(1)
                left = cond.split(operator)[0]
                right = cond.split(operator)[1]
                cols.add(left)
                conds.append(left)
                conds.append(operator)
                conds.append(right)
            print(tables, joins, conds, cardinality)
            output.append(','.join(tables) + '#' + ','.join(joins) + '#' + ','.join(conds) + '#' + cardinality)
    with open(sql_path + '.csv', 'w') as f:
        for line in output:
            f.write(line)
            f.write('\n')
    return cols


def prepare_samples(sql_path, samples):
    sample_bitmaps = []
    with open(sql_path + '.csv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            tables = [x.split(' ')[1] for x in line.split('#')[0].split(',')]
            conds = [x for x in line.split('#')[2].split(',')]
            table2conditions = {}
            for i in range(int(len(conds) / 3)):
                t = conds[i * 3].split('.')[0]
                attr = conds[i * 3].split('.')[1]
                op = conds[i * 3 + 1]
                value = conds[i * 3 + 2]
                if t in table2conditions:
                    table2conditions[t].append((attr, op, value))
                else:
                    table2conditions[t] = [(attr, op, value)]
            sample_bitmap = []
            for table in tables:
                try:
                    print(table2conditions)
                    print(table)
                    data_samples = samples[table]
                    conds = table2conditions[table]

                    bool_array = None
                    for cond in conds:
                        print(table, cond)
                        attr = cond[0]
                        if cond[1] == '=':
                            barray = (data_samples[attr] == float(cond[2]))
                        elif cond[1] == '<':
                            barray = (data_samples[attr] < float(cond[2]))
                        elif cond[1] == '>':
                            barray = (data_samples[attr] > float(cond[2]))
                        else:
                            raise Exception(cond)
                        if bool_array is None:
                            bool_array = barray
                        else:
                            bool_array = bool_array & barray
                    sample_bitmap.append(bool_array.astype(int).values)
                except Exception as e:
                    # f2.write('Pass '+query+'\n')
                    pass
                continue
            sample_bitmaps.append(sample_bitmap)
    return sample_bitmaps


def get_col_statistics(cols, data_dir):
    alias2table = {'ci': 'cast_info', 'mc': 'movie_companies', 'mi': 'movie_info', 'mk': 'movie_keyword',
                   'mi_idx': 'movie_info_idx', 't': 'title'}
    names = []
    cards = []
    distinct_nums = []
    mins = []
    maxs = []
    for col in cols:
        names.append(col)
        print(col)
        col_materialize = \
        pd.read_csv(data_dir + '/' + alias2table[col.split('.')[0]] + '.csv', quotechar='"', escapechar='\\',
                    error_bad_lines=False, low_memory=False)[col.split('.')[1]]
        maxs.append(col_materialize.max())
        mins.append(col_materialize.min())
        cards.append(len(col_materialize))
        distinct_nums.append(len(col_materialize.unique()))
    statistics = pd.DataFrame(
        data={'name': names, 'min': mins, 'max': maxs, 'cardinality': cards, 'num_unique_values': distinct_nums})
    statistics.to_csv(min_max_file, index=False)


def select_samples(data_dir):
    table2alias = {'cast_info': 'ci', 'movie_companies': 'mc', 'movie_info': 'mi', 'movie_keyword': 'mk',
                   'movie_info_idx': 'mi_idx', 'title': 't'}
    samples = {}
    for table, alias in table2alias.items():
        samples[alias] = pd.read_csv('{}/{}'.format(data_dir, table + '.csv'), quotechar='"', escapechar='\\',
                                     error_bad_lines=False, low_memory=False).sample(n=1000)
    return samples


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MSCN preprocessing')
    parser.add_argument('--datasets-dir', type=str, help='datasets_dir',
                        default='/home/jintao/CardinalityEstimationBenchmark/Distinct-Value-High/')
    parser.add_argument('--raw-query-file', type=str, help='sqls to be parsed',
                        default='/home/jintao/CardinalityEstimationBenchmark/train-test-data/cols-sql/4/train-4-num.sql')
    parser.add_argument('--min-max-file', type=str, help='Min Max',
                        default='/home/jintao/CardinalityEstimationBenchmark/learnedcardinalities-master/data/column_min_max_vals.csv')
    args = parser.parse_args()
    min_max_file = args.min_max_file
    sql_path = args.raw_query_file
    cols = preprocess_sql(sql_path)
    data_dir = args.datasets_dir
    get_col_statistics(cols, data_dir)
    if not os.path.exists(data_dir + '/samples.dict'):
        samples = select_samples(data_dir)
        with open(data_dir + '/samples.dict', 'wb') as f:
            pickle.dump(samples, f)
    else:
        with open(data_dir + '/samples.dict', 'rb') as f:
            samples = pickle.load(f)
    sample_bitmaps = prepare_samples(sql_path, samples)
    with open(sql_path + '.samplebitmap', 'wb') as f:
        pickle.dump(sample_bitmaps, f)
    print(sample_bitmaps[0])

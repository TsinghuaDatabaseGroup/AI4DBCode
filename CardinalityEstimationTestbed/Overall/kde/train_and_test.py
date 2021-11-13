import logging
import re
import time
from os import listdir

import numpy as np
import psycopg2
from scipy import stats
from tqdm import tqdm

L = logging.getLogger(__name__)


class FeedbackKDE():
    def __init__(self, database_conf, seed, sample_num, retrain, recollect, use_gpu):

        self.conn = psycopg2.connect(host=database_conf['host'],
                                     database=database_conf['database'],
                                     user=database_conf['user'])
        self.conn.set_session('read uncommitted', autocommit=True)
        self.cursor = self.conn.cursor()
        # Make sure that debug mode is deactivated and that all model traces are removed (unless we want to reuse the model):
        self.cursor.execute(f"SELECT setseed({1 / seed});")
        # self.cursor.execute("SET kde_debug TO true;")
        self.cursor.execute("SET kde_debug TO false;")
        if use_gpu:
            self.cursor.execute("SET ocl_use_gpu TO true;")
        else:
            self.cursor.execute("SET ocl_use_gpu TO false;")
        self.cursor.execute("SET kde_error_metric TO Quadratic;")

        # Remove all existing model traces if we don't reuse the model.
        if (retrain):
            self.cursor.execute("DELETE FROM pg_kdemodels;")
            self.cursor.execute("SELECT pg_stat_reset();")
        if (recollect):
            self.cursor.execute("DELETE FROM pg_kdemodels;")
            self.cursor.execute("DELETE FROM pg_kdefeedback;")
            self.cursor.execute("SELECT pg_stat_reset();")

        # KDE-specific parameters.
        self.cursor.execute(f"SET kde_samplesize TO {sample_num};")
        self.cursor.execute("SET kde_enable TO true;")
        self.cursor.execute("SET kde_collect_feedback TO true;")

        self.trained_table = set([])

    def transform(self, line):
        sql = ','.join(line.split(',')[:-1])
        cardinality = line.split(',')[-1].strip('\n')
        tables = [x.strip() for x in re.search('FROM(.*)WHERE', sql, re.IGNORECASE).group(1).split(',')]
        joins = [x.strip() for x in re.search('WHERE(.*)', sql, re.IGNORECASE).group(1).split('AND')[0:len(tables) - 1]]
        conditions = [x.strip(' ;\n') for x in
                      re.search('WHERE(.*)', sql, re.IGNORECASE).group(1).split('AND')[len(tables) - 1:]]
        conds = []
        for cond in conditions:
            operator = re.search('([<>=])', cond, re.IGNORECASE).group(1)
            left = cond.split(operator)[0]
            right = cond.split(operator)[1]
            conds.append(left)
            conds.append(operator)
            conds.append(right)
        query = ','.join(tables) + '#' + ','.join(joins) + '#' + ','.join(conds) + '#' + cardinality

        tables = query.split('#')[0].split(',')
        alias = [t.split(' ')[1] for t in tables]
        conditions = query.split('#')[2].split(',')
        cond = []
        if len(tables) > 1:
            for i in range(int(len(conditions) / 3)):
                cond.append(
                    conditions[i * 3].split('.')[0] + '_' + conditions[i * 3].split('.')[1] + conditions[i * 3 + 1] +
                    conditions[i * 3 + 2])
            self.trained_table.add('_'.join(sorted(alias)))
            return 'SELECT * FROM ' + '_'.join(sorted(alias)) + ' WHERE ' + ' AND '.join(cond) + ';'
        else:
            for i in range(int(len(conditions) / 3)):
                cond.append(conditions[i * 3].split('.')[1] + conditions[i * 3 + 1] + conditions[i * 3 + 2])
            self.trained_table.add(alias[0])
            return 'SELECT * FROM ' + tables[0] + ' WHERE ' + ' AND '.join(cond) + ';'  # alias[0]

    def train_batch(self, queries, single_data_dir, retrain, recollect):
        for i, query in tqdm(enumerate(queries)):
            if retrain:
                self.transform(query)
            if recollect:
                self.cursor.execute(self.transform(query))
            if (i + 1) % 100 == 0:
                L.info(f"{i + 1} queries done")
        L.info("Finishing running all training queries")
        start = time.time()
        self.cursor.execute("SET kde_collect_feedback TO false;")  # We don't need further feedback collection.
        self.cursor.execute("SET kde_enable_bandwidth_optimization TO true;")
        self.cursor.execute(f"SET kde_optimization_feedback_window TO {len(queries)};")

        for f in listdir(single_data_dir):
            if f.endswith('.csv'):
                table_name = f[:-4]
                if table_name in self.trained_table:
                    with open(f'{single_data_dir}/{f}', 'r') as ff:
                        columns = [x for x in ff.readline().strip().split(',')]
                    stat_cnt = 100
                    for c in columns:
                        self.cursor.execute(f"alter table {table_name} alter column {c} set statistics {stat_cnt};")
                    self.cursor.execute(f"analyze {table_name}({','.join(columns)});")
                    print(f"analyze {table_name}({','.join(columns)});")
                    self.conn.commit()
                    sample_file = f"/tmp/sample_{table_name}.csv"
                    self.cursor.execute(f"SELECT kde_dump_sample('{table_name}', '{sample_file}');")
        end = time.time()
        print(f'Training time: {end - start}s')

    def query(self, query):
        # key, isjoin = get_query_key(query)
        # if isjoin:
        #     total_num = self.pattern2totalnum[key]
        #     ratio = total_num / 600000.0
        # else:
        #     ratio = 1.0
        sql = f"explain(format json) {self.transform(query)}"
        start_stmp = time.time()
        self.cursor.execute(sql)
        dur_ms = (time.time() - start_stmp) * 1e3
        res = self.cursor.fetchall()
        card = res[0][0][0]['Plan']['Plan Rows']
        #  L.info(card)
        return card, dur_ms

    def oracle_query(self, query):
        sql = f"explain(ANALYZE true, format json) {self.transform(query)}"
        start_stmp = time.time()
        self.cursor.execute(sql)
        dur_ms = (time.time() - start_stmp) * 1e3
        res = self.cursor.fetchall()
        card = res[0][0][0]['Plan']['Actual Rows']
        #  L.info(card)
        return card, dur_ms


def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))


def print_mse(preds_unnorm, labels_unnorm):
    print("MSE: {}".format(((preds_unnorm - labels_unnorm) ** 2).mean()))


def print_mape(preds_unnorm, labels_unnorm):
    print("MAPE: {}".format(((np.abs(preds_unnorm - labels_unnorm) / labels_unnorm)).mean() * 100))


def print_pearson_correlation(x, y):
    PCCs = stats.pearsonr(x, y)
    print("Pearson Correlation: {}".format(PCCs))


def get_query_key(line):
    sql = ','.join(line.split(',')[:-1])
    cardinality = line.split(',')[-1].strip('\n')
    tables = [x.strip() for x in re.search('FROM(.*)WHERE', sql, re.IGNORECASE).group(1).split(',')]
    joins = [x.strip() for x in re.search('WHERE(.*)', sql, re.IGNORECASE).group(1).split('AND')[0:len(tables) - 1]]
    conditions = [x.strip(' ;\n') for x in
                  re.search('WHERE(.*)', sql, re.IGNORECASE).group(1).split('AND')[len(tables) - 1:]]
    conds = []
    for cond in conditions:
        operator = re.search('([<>=])', cond, re.IGNORECASE).group(1)
        left = cond.split(operator)[0]
        right = cond.split(operator)[1]
        conds.append(left)
        conds.append(operator)
        conds.append(right)
    query = ','.join(tables) + '#' + ','.join(joins) + '#' + ','.join(conds) + '#' + cardinality

    tables = query.split('#')[0].split(',')
    alias = [t.split(' ')[1] for t in tables]
    return '_'.join(sorted(alias)), len(tables) > 1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='KDE tester')
    parser.add_argument('--train-query-file', type=str)
    parser.add_argument('--test-query-file', type=str)
    parser.add_argument('--single-data-dir', type=str)
    parser.add_argument('--database', type=str)
    parser.add_argument('--sample-num', type=int)
    parser.add_argument('--train-num', type=int, default=10000)
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--recollect', action='store_true')
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--seed', type=float, default=1)

    args = parser.parse_args()
    # prioriy: params['version'] (build statistics from another dataset) > version (build statistics on the same dataset)
    database_conf = {'host': 'localhost', 'database': args.database, 'user': 'dodo'}
    with open(args.train_query_file, 'r') as f:
        queries = f.readlines()
        queries = queries[:min(args.train_num, len(queries))]

    L.info("construct postgres estimator...")
    estimator = FeedbackKDE(database_conf, args.seed, args.sample_num, args.retrain, args.recollect, args.use_gpu)

    L.info(f"start training with {len(queries)} queries...")
    start_stmp = time.time()
    estimator.train_batch(queries, args.single_data_dir, args.retrain, args.recollect)
    dur_min = (time.time() - start_stmp) / 60
    L.info(f"built kde estimator: {estimator}, using {dur_min:1f} minutes")

    with open(args.test_query_file, 'r') as f:
        test_queries = f.readlines()
        true_cards = []
        est_cards = []
        durs = []
        for query in test_queries:
            true_card = int(query.split(',')[-1].strip())
            # true_card, _ = estimator.oracle_query(query)
            if true_card > 0:
                est_card, dur_ms = estimator.query(query)
                true_cards.append(true_card)
                est_cards.append(est_card)
                durs.append(dur_ms)

    with open(args.test_query_file + '.kde.results.txt', 'w') as f:
        for i in range(len(est_cards)):
            f.write(f'{est_cards[i]},{true_cards[i]}')
            f.write('\n')
    print_qerror(np.array(est_cards), np.array(true_cards))
    print_mse(np.array(est_cards), np.array(true_cards))
    print_mape(np.array(est_cards), np.array(true_cards))
    print_pearson_correlation(np.array(est_cards), np.array(true_cards))
    print(f'test time {np.mean(durs)}ms')

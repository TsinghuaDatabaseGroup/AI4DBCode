import argparse
import os
import pickle

import numpy as np
from scipy import stats

parser = argparse.ArgumentParser(description='Quick Sel Preprocessing')
parser.add_argument('--testpath', type=str, help='sqls to be parsed',
                    default='../../../../../../train-test-data/cols-sql/4/test-only4-num.sql')
args = parser.parse_args()


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


if __name__ == '__main__':
    est = []
    true = []
    path = './JOB/' + args.testpath
    with open('../../../../../../pattern2totalnum.pkl', 'rb') as f:
        pattern2totalnum = pickle.load(f)
        print('pattern2totalnum:', pattern2totalnum)
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            with open(path + filename, 'r') as f:
                key = '_'.join(filename.split('_')[:-1])
                print(pattern2totalnum[key])
                for x in f.readlines():
                    est.append(max(float(x.split(',')[0]) * pattern2totalnum[key], 1.0))
                    true.append(max(float(x.split(',')[1]) * pattern2totalnum[key], 1.0))
    est = np.array(est)
    true = np.array(true)

    print_qerror(est, true)
    print_mse(est, true)
    print_mape(est, true)
    print_pearson_correlation(est, true)

    a = true.tolist()
    b = est.tolist()
    c = []
    c.append(a)
    c.append(b)
    c = np.array(c)
    c = np.rot90(c)
    c = np.rot90(c)
    c = np.rot90(c)
    # np.savetxt('../../../../../../train-test-data/cols-sql/4/' +  'quicksel.result.csv', c, delimiter = ',')

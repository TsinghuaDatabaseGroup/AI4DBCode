import xgboost as xgb
import torch
from torch import nn
from torch import optim

import pandas as pd
import numpy as np
import pickle
from scipy import stats
import math
import time

import argparse

parser = argparse.ArgumentParser(description='Local NN')
parser.add_argument('--train-file', type=str, help='datasets_dir',
                    default='/home/jintao/CardinalityEstimationBenchmark/train-test-data/cols-sql/4/train-4-num.sql')
parser.add_argument('--test-file', type=str, help='sqls to be parsed',
                    default='/home/jintao/CardinalityEstimationBenchmark/train-test-data/cols-sql/4/test-only4-num.sql')
parser.add_argument('--min-max-file', type=str, help='Min Max',
                    default='/home/jintao/CardinalityEstimationBenchmark/learnedcardinalities-master/data/column_min_max_vals.csv')
parser.add_argument('--model', type=str, help='nn or xgb', default='nn')
parser.add_argument("--version", help="version", type=str, default='cols_4_distinct_1000_corr_5_skew_5')

args = parser.parse_args()
min_max_file = args.min_max_file
# min_max_file = '/home/jintao/CardinalityEstimationBenchmark/learnedcardinalities-master/data/column_min_max_vals.csv'
fmetric = open('../metric_result/' + args.version + '.' + args.model + '.txt', 'w')


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.module(x)


class NNNet():
    def __init__(self, input_dim):
        self.model = MLP(input_dim, 128)

    def train(self, train_data, labels, num_round=10):
        batch_size = 64
        learning_rate = 0.01
        print(train_data.shape, labels.shape)
        train_len = int(0.8 * len(train_data))
        training_data = torch.FloatTensor(train_data[:train_len])
        training_label = torch.FloatTensor(labels[:train_len]).unsqueeze(1)
        validate_data = torch.FloatTensor(train_data[train_len:])
        validate_label = torch.FloatTensor(labels[train_len:]).unsqueeze(1)
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(training_data, training_label),
                                                   batch_size=batch_size, shuffle=True)
        validate_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(validate_data, validate_label),
                                                      batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        Loss = nn.MSELoss()
        self.model.train()
        for epoch in range(num_round):
            for batch_idx, (data, target) in enumerate(train_loader):
                logits = self.model(data)
                loss = Loss(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('Train Epoch: {} {} \tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), loss.item()))
        self.model.eval()
        test_loss = 0
        for data, target in validate_loader:
            logits = self.model(data)
            test_loss += Loss(logits, target).item()
        test_loss /= len(validate_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, {} \n'.format(
            test_loss, len(validate_loader.dataset)))

    def estimate(self, test_data):
        test_data = torch.FloatTensor(test_data)
        logits = self.model(test_data)
        predicts = []
        predicts += logits.squeeze(1).tolist()
        return np.array(predicts)


class TreeEnsemble():

    def __init__(self):
        self.model = xgb.Booster({'nthread': 10})

    def train(self, train_data, labels, num_round=10,
              param={'max_depth': 5, 'eta': 0.1, 'booster': 'gbtree', 'objective': 'reg:logistic'}):
        print(train_data.shape, labels.shape)
        train_len = int(0.8 * len(train_data))
        dtrain = xgb.DMatrix(train_data[:train_len], label=labels[:train_len])
        dvalidate = xgb.DMatrix(train_data[train_len:], label=labels[train_len:])
        evallist = [(dvalidate, 'test'), (dtrain, 'train')]
        self.model = xgb.train(param, dtrain, num_round, evallist)

    def save_model(self, path):
        self.model.save_model(path + '.xgb.model')

    def load_model(self, path):
        self.model.load_model(path + '.xgb.model')

    def estimate(self, test_data):
        dtest = xgb.DMatrix(test_data)
        return self.model.predict(dtest)


def normalize(x, min_card_log, max_card_log):
    return np.maximum(np.minimum((np.log(x) - min_card_log) / (max_card_log - min_card_log), 1.0), 0.0)


def unnormalize(x, min_card_log, max_card_log):
    return np.exp(x * (max_card_log - min_card_log) + min_card_log)


def prepare_pattern_workload(path):
    pattern2training = {}
    pattern2truecard = {}
    minmax = pd.read_csv(min_max_file)
    minmax = minmax.set_index('name')
    min_card_log = 999999999999.0
    max_card_log = 0.0
    with open(path + '.csv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            tables = sorted([x.split(' ')[1] for x in line.split('#')[0].split(',')])
            local_cols = []
            vecs = []
            for col_name in minmax.index:
                if col_name.split('.')[0] in tables:
                    local_cols.append(col_name)
                    vecs.append(0.0)
                    vecs.append(1.0)
            conds = [x for x in line.split('#')[2].split(',')]
            for i in range(int(len(conds) / 3)):
                attr = conds[i * 3]
                op = conds[i * 3 + 1]
                value = conds[i * 3 + 2]
                idx = local_cols.index(attr)
                maximum = float(minmax.loc[attr]['max'])
                minimum = float(minmax.loc[attr]['min'])
                distinct_num = minmax.loc[attr]['num_unique_values']
                if op == '=':
                    offset = (maximum - minimum) / distinct_num / 2.0
                    upper = ((float(value) + offset) - minimum) / (maximum - minimum)
                    lower = (float(value) - offset - minimum) / (maximum - minimum)
                elif op == '<':
                    upper = (float(value) - minimum) / (maximum - minimum)
                    lower = 0.0
                elif op == '>':
                    upper = 1.0
                    lower = (float(value) - minimum) / (maximum - minimum)
                else:
                    raise Exception(op)
                if upper < vecs[idx * 2 + 1]:
                    vecs[idx * 2 + 1] = upper
                if lower > vecs[idx * 2]:
                    vecs[idx * 2] = lower
            key = '_'.join(tables)
            card = float(line.split('#')[-1])
            if key in pattern2training:
                pattern2training[key].append(vecs)
                pattern2truecard[key].append(card)
            else:
                pattern2training[key] = [vecs]
                pattern2truecard[key] = [card]
            if math.log(card) < min_card_log:
                min_card_log = math.log(card)
            if math.log(card) > max_card_log:
                max_card_log = math.log(card)

    return pattern2training, pattern2truecard, min_card_log, max_card_log


def train_for_all_pattern(path, t='xgb'):
    pattern2training, pattern2truecard, min_card_log, max_card_log = prepare_pattern_workload(path)
    print('min_card_log: {}, max_card_log: {}'.format(min_card_log, max_card_log))
    pattern2model = {}
    for k, v in pattern2training.items():
        print(k, len(v), len(v[0]))
        print(v[0])
        print(v[1])
        if t == 'xgb':
            pattern2model[k] = TreeEnsemble()
        elif t == 'nn':
            pattern2model[k] = NNNet(len(v[0]))
        pattern2model[k].train(np.array(v), normalize(pattern2truecard[k], min_card_log, max_card_log), num_round=100)
    with open(path + '.{}.model'.format(t), 'wb') as f:
        pickle.dump(pattern2model, f)
    return pattern2model


def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    fmetric.write("Median: {}".format(np.median(qerror)) + '\n' + "90th percentile: {}".format(
        np.percentile(qerror, 90)) + '\n' + "95th percentile: {}".format(np.percentile(qerror, 95)) + \
                  '\n' + "99th percentile: {}".format(np.percentile(qerror, 99)) + '\n' + "99th percentile: {}".format(
        np.percentile(qerror, 99)) + '\n' + "Max: {}".format(np.max(qerror)) + '\n' + \
                  "Mean: {}".format(np.mean(qerror)) + '\n')
    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))


def print_mse(preds_unnorm, labels_unnorm):
    fmetric.write("MSE: {}".format(((preds_unnorm - labels_unnorm) ** 2).mean()) + '\n')
    print("MSE: {}".format(((preds_unnorm - labels_unnorm) ** 2).mean()))


def print_mape(preds_unnorm, labels_unnorm):
    fmetric.write("MAPE: {}".format(((np.abs(preds_unnorm - labels_unnorm) / labels_unnorm)).mean() * 100) + '\n')
    print("MAPE: {}".format(((np.abs(preds_unnorm - labels_unnorm) / labels_unnorm)).mean() * 100))


def print_pearson_correlation(x, y):
    PCCs = stats.pearsonr(x, y)
    fmetric.write("Pearson Correlation: {}".format(PCCs) + '\n')
    print("Pearson Correlation: {}".format(PCCs))


def test_for_all_pattern(path, model_name, pattern2model):
    pattern2testing, pattern2truecard, min_card_log, max_card_log = prepare_pattern_workload(path)
    print('min_card_log: {}, max_card_log: {}'.format(min_card_log, max_card_log))
    cards = []
    true_cards = []
    start = time.time()
    for k, v in pattern2testing.items():
        model = pattern2model[k]
        cards += unnormalize(model.estimate(np.array(v)), min_card_log, max_card_log).tolist()
        true_cards += pattern2truecard[k]
    end = time.time()
    fmetric.write(
        "Prediction Time {}ms for each of {} queries".format((end - start) / len(cards) * 1000, len(cards)) + '\n')
    print("Prediction Time {}ms for each of {} queries".format((end - start) / len(cards) * 1000, len(cards)))
    print_qerror(np.array(cards), np.array(true_cards))
    print_mse(np.array(cards), np.array(true_cards))
    print_mape(np.array(cards), np.array(true_cards))
    print_pearson_correlation(np.array(cards), np.array(true_cards))
    with open(f'{path}.{model_name}.results.csv', 'w') as f:
        for i in range(len(cards)):
            f.write(f'{cards[i]},{true_cards[i]}')
            f.write('\n')


if __name__ == '__main__':
    tt1 = time.time()
    pattern2model = train_for_all_pattern(args.train_file, args.model)
    fmetric.write("traintime:" + str(time.time() - tt1) + 's')
    print("traintime:", time.time() - tt1, 's')
    test_for_all_pattern(args.test_file, args.model, pattern2model)
    fmetric.close()

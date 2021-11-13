import math
import time

import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description='KDE')
parser.add_argument('--train-file', type=str, help='datasets_dir',
                    default='/home/jintao/CardinalityEstimationBenchmark/train-test-data/cols-sql/4/train-4-num.sql')
parser.add_argument('--test-file', type=str, help='sqls to be parsed',
                    default='/home/jintao/CardinalityEstimationBenchmark/train-test-data/cols-sql/4/test-only4-num.sql')
parser.add_argument('--min-max-file', type=str, help='Min Max',
                    default='/home/jintao/CardinalityEstimationBenchmark/learnedcardinalities-master/data/column_min_max_vals.csv')
parser.add_argument("--version", help="version", type=str, default='cols_4_distinct_1000_corr_5_skew_5')

args = parser.parse_args()
min_max_file = args.min_max_file
trainpath = args.train_file
testpath = args.test_file
version = args.version

class KDE():
    def __init__(self, samples, num_attributes):
        self.H = torch.rand(num_attributes)
        self.H = Variable(self.H, requires_grad=True)
        self.sample = samples
        self.num_attributes = num_attributes

    def train(self, train_predicates, cardinalities, total_card, batch_size, num_epochs):
        num_sample = len(self.sample)
        predicates_lower = torch.FloatTensor([[pre[i] for i in range(0, len(pre), 2)] for pre in train_predicates])
        predicates_upper = torch.FloatTensor([[pre[i] for i in range(1, len(pre), 2)] for pre in train_predicates])
        cardinalities = (torch.FloatTensor(cardinalities)) / total_card
        samples = torch.FloatTensor(self.sample)
        optimizer = torch.optim.Adam([self.H], lr=0.001)
        total_size = len(train_predicates)
        split_idx = int(0.8 * total_size)
        train_dataset = TensorDataset(predicates_lower[:split_idx], predicates_upper[:split_idx],
                                      cardinalities[:split_idx])
        validate_dataset = TensorDataset(predicates_lower[split_idx:], predicates_upper[split_idx:],
                                         cardinalities[split_idx:])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)
        train_start_time = time.time()
        for epoch in range(num_epochs):
            print('Epoch:', epoch)
            for i, (lower, upper, targets) in enumerate(train_loader, 0):
                optimizer.zero_grad()
                ress = 0.0
                for sample in samples:
                    res = 1.0
                    for j in range(self.H.shape[0]):
                        res *= (torch.erf((upper[:, j] - sample[j]) / (math.sqrt(2) * self.H[j])) - torch.erf(
                            (lower[:, j] - sample[j]) / (math.sqrt(2) * self.H[j])))
                    ress += res / (2 ** num_attributes)
                # print('ress:', ress)
                # print('targets:', targets)
                ress = ress / (total_card/num_sample)  # modify
                loss = torch.nn.functional.mse_loss(ress, targets)
                print(f'Training Loss: {loss*(total_card/num_sample)}')
                loss.backward()
                optimizer.step()
        train_end_time = time.time()
        print(f'Training Time: {train_end_time - train_start_time}s')

        mses = []
        qerrors = []
        for i, (lower, upper, targets) in enumerate(validate_loader, 0):
            ress = 0.0
            for sample in samples:
                res = 1.0
                for j in range(self.H.shape[0]):
                    res *= (torch.erf((upper[:, j] - sample[j]) / (math.sqrt(2) * self.H[j])) - torch.erf(
                        (lower[:, j] - sample[j]) / (math.sqrt(2) * self.H[j])))
                ress += res / (2 ** num_attributes)
                ress = ress / (total_card/num_sample)
            mse = (ress - targets) ** 2
            qerror = torch.max(ress, targets) / torch.max(torch.zeros_like(ress) + 1e-5,
                                                          (torch.min(ress, targets) + 1e-5))
            mses.append(mse)
            qerrors.append(qerror)
        qerrors = torch.cat(qerrors).detach().numpy()
        # mses = torch.cat(mses).detach().numpy()
        print(f'Validate Mean Q-error: {qerrors.mean()}')
        print(f'Validate 50th Q-error: {np.median(qerrors)}')
        print(f'Validate 90th Q-error: {np.percentile(qerrors, 90)}')
        print(f'Validate 95th Q-error: {np.percentile(qerrors, 95)}')
        print(f'Validate 99th Q-error: {np.percentile(qerrors, 99)}')
        print(f'Validate 100th Q-error: {qerrors.max()}')
        # print(f'Validate MSE: {mses.mean()}')

    def test(self, test_predicates, cardinalities, total_card):
        num_sample = len(self.sample)
        predicates_lower = torch.FloatTensor([[pre[i] for i in range(0, len(pre), 2)] for pre in test_predicates])
        predicates_upper = torch.FloatTensor([[pre[i] for i in range(1, len(pre), 2)] for pre in test_predicates])
        cardinalities = (torch.FloatTensor(cardinalities)) / total_card
        samples = torch.FloatTensor(self.sample)
        test_dataset = TensorDataset(predicates_lower, predicates_upper, cardinalities)
        test_loader = torch.utils.data.DataLoader(test_dataset)
        start_time = time.time()
        mses = []
        qerrors = []
        for i, (lower, upper, targets) in enumerate(test_loader, 0):
            ress = 0.0
            print(i)
            for sample in samples:
                res = 1.0
                for j in range(self.H.shape[0]):
                    res *= (torch.erf((upper[:, j] - sample[j]) / (math.sqrt(2) * self.H[j])) - torch.erf(
                        (lower[:, j] - sample[j]) / (math.sqrt(2) * self.H[j])))
                ress += res / (2 ** num_attributes)
                ress = ress / (total_card/num_sample)
            mse = (ress - targets) ** 2
            qerror = torch.max(ress, targets) / torch.max(torch.zeros_like(ress) + 1e-5,
                                                          (torch.min(ress, targets) + 1e-5))
            mses.append(mse)
            qerrors.append(qerror)
        end_time = time.time()
        print(f'Test Time: {end_time - start_time}')
        qerrors = torch.cat(qerrors).detach().numpy()
        # mses = torch.cat(mses).detach().numpy()
        print(f'Test Mean Q-error: {qerrors.mean()}')
        print(f'Test 50th Q-error: {np.median(qerrors)}')
        print(f'Test 90th Q-error: {np.percentile(qerrors, 90)}')
        print(f'Test 95th Q-error: {np.percentile(qerrors, 95)}')
        print(f'Test 99th Q-error: {np.percentile(qerrors, 99)}')
        # print(f'MSE: {mses.mean()}')

def prepare_pattern_workload(path):
    Embed = []
    truecard = []
    minmax = pd.read_csv(min_max_file)
    minmax = minmax.set_index('name')
    min_card = 999999999999.0
    max_card = 0.0
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

            card = float(line.split('#')[-1])
            Embed.append(vecs)
            truecard.append(card)
            '''
            if card < min_card:
                min_card = math.log(card)
            if card > max_card:
                max_card = math.log(card)
            '''
    num_attributes = int(len(vecs)/2)
    return Embed, truecard, num_attributes


# if __name__ == '__name__':
# 0.4 <= B <= 0.5
# min(A) <= A <= max(A)
# sql parser

batch_size = 256
num_epochs = 10
total_card = 100000
train_predicates, train_cardinalities, num_attributes = prepare_pattern_workload(trainpath)
table = pd.read_csv('../csvdata_sql/' + version + '.csv').apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
sample = table.sample(n=2000).values.tolist()
kde = KDE(samples=sample, num_attributes = num_attributes)
kde.train(train_predicates, train_cardinalities, total_card, batch_size, num_epochs)
# kde.train(train_predicates, train_cardinalities, total_card, batch_size, num_epochs)

test_predicates, test_cardinalities, num_attributes = prepare_pattern_workload(testpath)
kde.test(test_predicates, test_cardinalities, total_card)

'''train_predicates = [[0.1, 0.3, 0.4, 0.5],
                    [0.3, 0.6, 0.7, 0.8],
                    [0.1, 0.4, 0.9, 1.0],
                    [0.2, 0.3, 0.7, 0.8],
                    [0.4, 0.6, 0.5, 0.6]]
test_predicates = [[0.1, 0.3, 0.5, 0.7],
                    [0.1, 0.4, 0.6, 0.7]]
train_cardinalities = [20, 30, 90, 15, 31]
test_cardinalities = [30, 40]
num_sample = 6
# sample from data sampler
sample = [[0.15, 0.35], [0.35, 0.55], [0.45, 0.25], [0.65, 0.25], [0.85, 0.15], [0.55, 0.78]]
num_attributes = 2
num_epochs = 1000
batch_size = 156
total_card = 200  # rows of table

kde = KDE(samples=sample, num_attributes=num_attributes)
kde.train(train_predicates, train_cardinalities, total_card, batch_size, num_epochs)
kde.test(test_predicates, test_cardinalities, total_card)
    '''

import argparse
import os
import time

import torch
from mscn.data import get_train_datasets, load_data, make_dataset
from mscn.model import SetConv
from mscn.util import *
from scipy import stats
from torch.autograd import Variable
from torch.utils.data import DataLoader

# min_max_file = '/home/jintao/CardinalityEstimationBenchmark/learnedcardinalities-master/data/column_min_max_vals.csv'
parser = argparse.ArgumentParser(description='MSCN.')
parser.add_argument('--min-max-file', type=str, help='Min Max',
                    default='/home/jintao/CardinalityEstimationBenchmark/learnedcardinalities-master/data/column_min_max_vals.csv')
parser.add_argument("--queries", help="number of training queries (default: 10000)", type=int, default=10000)
parser.add_argument("--epochs", help="number of epochs (default: 10)", type=int, default=10)
parser.add_argument("--batch", help="batch size (default: 1024)", type=int, default=1024)
parser.add_argument("--hid", help="number of hidden units (default: 256)", type=int, default=256)
parser.add_argument("--train", help="need train", action='store_true')
parser.add_argument("--cuda", help="use CUDA", action="store_true")
parser.add_argument("--version", help="version", type=str, default='cols_4_distinct_1000_corr_5_skew_5')
parser.add_argument("--train-query-file", help="train queries (no suffix)",
                    default='/home/jintao/CardinalityEstimationBenchmark/train-test-data/cols-sql/2/train-2-num.sql')
parser.add_argument("--test-query-file", help="train queries (no suffix)",
                    default='/home/jintao/CardinalityEstimationBenchmark/train-test-data/cols-sql/2/test-2-num.sql')
args = parser.parse_args()
print(args.queries, args.epochs, args.batch, args.hid, args.cuda, args.train, args.min_max_file)
# global min_max_file  # quanju
min_max_file = args.min_max_file


# fmetric = open('/home/zhangjintao/CardBenchmark-Revision/Update/updated-data/metric/' + args.version + '.update_mscn.txt', 'a')


def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)


def qerror_loss(preds, targets, min_val, max_val):
    qerror = []
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)

    for i in range(len(targets)):
        if (preds[i] > targets[i]).cpu().data.numpy()[0]:
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
    return torch.mean(torch.cat(qerror))


def predict(model, data_loader, cuda):
    preds = []
    t_total = 0.

    model.eval()
    for batch_idx, data_batch in enumerate(data_loader):

        samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch

        if cuda:
            samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
            sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
        samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(joins), Variable(
            targets)
        sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(
            join_masks)

        t = time.time()
        outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
        t_total += time.time() - t

        for i in range(outputs.data.shape[0]):
            preds.append(outputs.data[i])

    return preds, t_total


def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    '''fmetric.write("Median: {}".format(np.median(qerror))+ '\n'+ "90th percentile: {}".format(np.percentile(qerror, 90))+ '\n'+ "95th percentile: {}".format(np.percentile(qerror, 95))+\
            '\n'+ "99th percentile: {}".format(np.percentile(qerror, 99))+ '\n'+ "99th percentile: {}".format(np.percentile(qerror, 99))+ '\n'+ "Max: {}".format(np.max(qerror))+ '\n'+\
            "Mean: {}".format(np.mean(qerror))+ '\n')'''

    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))


def print_mse(preds_unnorm, labels_unnorm):
    # fmetric.write("MSE: {}".format(((preds_unnorm - labels_unnorm) ** 2).mean())+ '\n')
    print("MSE: {}".format(((preds_unnorm - labels_unnorm) ** 2).mean()))


def print_mape(preds_unnorm, labels_unnorm):
    # fmetric.write("MAPE: {}".format(((np.abs(preds_unnorm - labels_unnorm) / labels_unnorm)).mean() * 100)+ '\n')
    print("MAPE: {}".format(((np.abs(preds_unnorm - labels_unnorm) / labels_unnorm)).mean() * 100))


def print_pearson_correlation(x, y):
    PCCs = stats.pearsonr(x, y)
    # fmetric.write("Pearson Correlation: {}".format(PCCs)+ '\n\n')
    print("Pearson Correlation: {}".format(PCCs))


def train_and_predict(train_file, test_file, num_queries, num_epochs, batch_size, hid_units, cuda, need_train=True):
    # Load training and validation data
    print(min_max_file)
    num_materialized_samples = 1000
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_data, test_data = get_train_datasets(
        num_queries, num_materialized_samples, train_file, min_max_file)
    table2vec, column2vec, op2vec, join2vec = dicts
    print('need_train: ', need_train)
    print('train_file: ', train_file)
    print('test_file: ', train_file)
    sample_feats = len(table2vec) + num_materialized_samples
    predicate_feats = len(column2vec) + len(op2vec) + 1
    join_feats = len(join2vec)

    # if need_train:
    # Train model

    model = SetConv(sample_feats, predicate_feats, join_feats, hid_units)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if cuda:
        model.cuda()

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)
    train_start = time.time()
    path = train_file + '.mscn.model'
    # model = SetConv(sample_feats, predicate_feats, join_feats, hid_units)
    # model.train()
    model.load_state_dict(torch.load(path))

    for epoch in range(num_epochs):
        model.train()
        if epoch != 0:

            loss_total = 0.

            for batch_idx, data_batch in enumerate(train_data_loader):

                samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch

                if cuda:
                    samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
                    sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
                samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(
                    joins), Variable(
                    targets)
                sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(
                    join_masks)

                optimizer.zero_grad()
                outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
                loss = qerror_loss(outputs, targets.float(), min_val, max_val)
                loss_total += loss.item()
                loss.backward()
                optimizer.step()

            print("Epoch {}, loss: {}".format(epoch, loss_total / len(train_data_loader)))
            train_end = time.time()

            # print('epoch' + str(epoch) + '\n' + "Training Time: {}s".format(train_end - train_start))
            # fmetric.write('\nepoch' + str(epoch) + ':\n'+"Training Time: {}s".format(train_end - train_start)+ '\n')  # 写入训练时间

            # torch.save(model.state_dict(), path)
        '''
        # Get final training and validation set predictions
        preds_train, t_total = predict(model, train_data_loader, cuda)
        print("Prediction time per training sample: {}".format(t_total / len(labels_train) * 1000))

        preds_test, t_total = predict(model, test_data_loader, cuda)
        print("Prediction time per validation sample: {}".format(t_total / len(labels_test) * 1000))

        # Unnormalize
        preds_train_unnorm = unnormalize_labels(preds_train, min_val, max_val)
        labels_train_unnorm = unnormalize_labels(labels_train, min_val, max_val)

        preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)
        labels_test_unnorm = unnormalize_labels(labels_test, min_val, max_val)

        path = train_file + '.mscn.model'
        torch.save(model.state_dict(), path)

        # Print metrics
        print("\nQ-Error training set:")
        print_qerror(preds_train_unnorm, labels_train_unnorm)
        print("\nMSE training set:")
        print_mse(preds_train_unnorm, labels_train_unnorm)
        print("\nMAPE training set:")
        print_mape(preds_train_unnorm, labels_train_unnorm)
        print("\nPearson Correlation training set:")
        print_pearson_correlation(preds_train_unnorm, labels_train_unnorm)

        print("\nQ-Error validation set:")
        print_qerror(preds_test_unnorm, labels_test_unnorm)
        print("\nMSE validation set:")
        print_mse(preds_test_unnorm, labels_test_unnorm)
        print("\nMAPE validation set:")
        print_mape(preds_test_unnorm, labels_test_unnorm)
        print("\nPearson Correlation validation set:")
        print_pearson_correlation(preds_test_unnorm, labels_test_unnorm)
        print("")
        '''
        # else:
        # model = SetConv(sample_feats, predicate_feats, join_feats, hid_units)
        # path = train_file+'.mscn.model'
        # model.load_state_dict(torch.load(path))
        model.eval()

        # Load test data
        file_name = test_file
        joins, predicates, tables, samples, label = load_data(file_name, num_materialized_samples)

        # Get feature encoding and proper normalization
        samples_test = encode_samples(tables, samples, table2vec)
        predicates_test, joins_test = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
        labels_test, _, _ = normalize_labels(label, min_val, max_val)

        print("Number of test samples: {}".format(len(labels_test)))

        max_num_predicates = max([len(p) for p in predicates_test])
        max_num_joins = max([len(j) for j in joins_test])

        # Get test set predictions
        test_data = make_dataset(samples_test, predicates_test, joins_test, labels_test, max_num_joins,
                                 max_num_predicates)
        test_data_loader = DataLoader(test_data, batch_size=batch_size)

        preds_test, t_total = predict(model, test_data_loader, cuda)
        # fmetric.write("Prediction time per test sample: {}ms".format(t_total / len(labels_test) * 1000)+ '\n')
        # fmetric.close()
        # print("Prediction time per test sample: {}ms".format(t_total / len(labels_test) * 1000))

        # Unnormalize
        preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)

        # Print metrics
        print("\nQ-Error " + test_file + ":")
        print_qerror(preds_test_unnorm, np.array(label, dtype=np.float64))
        # print("\nMSE validation set:")
        # print_mse(preds_test_unnorm, np.array(label, dtype=np.float64))
        # print("\nMAPE validation set:")
        # print_mape(preds_test_unnorm, np.array(label, dtype=np.float64))
        # print("\nPearson Correlation validation set:")
        # print_pearson_correlation(preds_test_unnorm, np.array(label, dtype=np.float64))

        # Write predictions
        file_name = test_file + ".mscn" + '_epoch' + str(epoch) + ".result.csv"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w") as f:
            for i in range(len(preds_test_unnorm)):
                f.write(str(preds_test_unnorm[i]) + "," + label[i] + "\n")
        f.close()  # remark


def main():
    train_and_predict(args.train_query_file, args.test_query_file, args.queries, args.epochs, args.batch, args.hid,
                      args.cuda, args.train)


if __name__ == "__main__":
    main()

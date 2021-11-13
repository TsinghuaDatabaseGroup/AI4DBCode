import sys

sys.path.append('../../src/python')
from quickSel import *

query_dir = '../../test/python/queries/'

data_file_name = query_dir + 'gaussian_3d.data'
query_file_name = query_dir + 'gaussian_trans.query'

qstep = 10
isomer_budget = 10
crumbs_budget = 300


# get data
def get_data():
    data_file = open(data_file_name)
    data = []
    for line in data_file.readlines():
        data.append([float(x) for x in line.split(',')])
    data_file.close()
    return data


# get queries
def get_queries():
    query_file = open(query_file_name)
    queries = []
    for line in query_file.readlines():
        [strll, strur, strfreq] = line.split(';')
        ll = [float(x) for x in strll.split(',')]
        ur = [float(x) for x in strur.split(',')]
        freq = float(strfreq)
        queries.append([{0: (ll[0], ur[0]),
                         1: (ll[1], ur[1]),
                         2: (ll[2], ur[2])},
                        freq])
    return queries


def count_tuple(data, boundary):
    assert (len(boundary) == 6)
    D = data
    D = [x for x in D if x[0] >= boundary[0] and x[0] <= boundary[3]]
    D = [x for x in D if x[1] >= boundary[1] and x[1] <= boundary[4]]
    D = [x for x in D if x[2] >= boundary[2] and x[2] <= boundary[5]]
    return len(D)


# generate permanent assertions
def generate_permanent_assertions(data):
    nx = 10
    ny = 10
    nz = 10
    xstep = 1.0 / nx
    ystep = 1.0 / ny
    zstep = 1.0 / nz
    boundaries = []

    for i in range(nx - 1):
        xstart = xstep * i
        xend = xstep * (i + 1)
        boundaries.append([xstart, 0.0, 0.0, xend, 1.0, 1.0])
    for i in range(ny - 1):
        ystart = ystep * i
        yend = ystep * (i + 1)
        boundaries.append([0.0, ystart, 0.0, 1.0, yend, 1.0])
    for i in range(nz - 1):
        zstart = zstep * i
        zend = zstep * (i + 1)
        boundaries.append([0.0, 0.0, zstart, 1.0, 1.0, zend])
    xstart = xstep * (nx - 1)
    xend = xstep * nx
    boundaries.append([xstart, 0.0, 0.0, xend, 1.0, 1.0])
    N = float(len(data))
    freqs = map(lambda b: float(count_tuple(data, b)) / N, boundaries)

    assertions = []
    for i in range(len(boundaries)):
        assertions.append(Assertion({0: (boundaries[i][0], boundaries[i][3]),
                                     1: (boundaries[i][1], boundaries[i][4]),
                                     2: (boundaries[i][2], boundaries[i][5])},
                                    freqs[i]))
    return assertions


# test quickSel
def test_crumbs(data, queries):
    perma_assertions = generate_permanent_assertions(data)
    output_file = open(query_dir + '../crumbs_output', 'w')
    output_file.truncate()
    for i in range(300):
        print
        i
        test_queries = [x[0] for x in queries[qstep * (i + 1): qstep * (i + 2)]]
        test_truth = [x[1] for x in queries[qstep * (i + 1): qstep * (i + 2)]]
        train_l = max(0, qstep * (i + 1) - crumbs_budget)
        train_r = qstep * (i + 1)
        new_assertions = [Assertion(query[0], query[1]) for query in queries[train_l: train_r]]
        min_max = Hyperrectangle([[0, 1], [0, 1], [0, 1]])
        start = time.time()
        quickSel = Crumbs(min_max)
        A, b, x = quickSel.assign_optimal_freq(perma_assertions + new_assertions)
        end = time.time()
        answers = map(lambda t: quickSel.answer(t), test_queries)
        err = []
        rel_err = []
        for j in range(len(answers)):
            err.append(abs(answers[j] - test_truth[j]))
            rel_err.append(abs(answers[j] - test_truth[j]) / max(100.0 / len(data), test_truth[j]))
            print
            str(answers[j]) + " " + str(test_truth[j])
        avg_rel_err = sum(rel_err) / len(rel_err)
        output_file.write(str(avg_rel_err) + '\n')
    output_file.close()


# test isomer
def test_isomer(data, queries):
    perma_assertions = generate_permanent_assertions(data)
    output_file = open(query_dir + '../isomer_output', 'w')
    output_file.truncate()
    for i in range(300):
        print
        i
        test_queries = [x[0] for x in queries[qstep * (i + 1): qstep * (i + 2)]]
        test_truth = [x[1] for x in queries[qstep * (i + 1): qstep * (i + 2)]]
        train_l = max(0, qstep * (i + 1) - isomer_budget)
        train_r = qstep * (i + 1)
        new_assertions = [Assertion(query[0], query[1]) for query in queries[train_l: train_r]]
        min_max = Hyperrectangle([[0, 1], [0, 1], [0, 1]])
        start = time.time()
        isomer = Isomer(min_max)
        isomer.assign_optimal_freq(perma_assertions + new_assertions)
        end = time.time()
        answers = map(lambda t: isomer.answer(t), test_queries)
        err = []
        rel_err = []
        for j in range(len(answers)):
            err.append(abs(answers[j] - test_truth[j]))
            rel_err.append(abs(answers[j] - test_truth[j]) / max(100.0 / len(data), test_truth[j]))
            print
            str(answers[j]) + " " + str(test_truth[j])
        avg_rel_err = sum(rel_err) / len(rel_err)
        output_file.write(str(avg_rel_err) + '\n')
    output_file.close()


data = get_data()
queries = get_queries()
test_crumbs(data, queries)
test_isomer(data, queries)

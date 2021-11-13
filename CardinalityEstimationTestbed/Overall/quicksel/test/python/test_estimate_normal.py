import numpy as np
from test_include import *

'''
# of queries: 5, number of bins in ISOMER: 7
# of queries: 10, number of bins in ISOMER: 30
# of queries: 15, number of bins in ISOMER: 80
# of queries: 20, number of bins in ISOMER: 203
# of queries: 25, number of bins in ISOMER: 603
# of queries: 30, number of bins in ISOMER: 1363
# of queries: 35, number of bins in ISOMER: 2640
# of queries: 40, number of bins in ISOMER: 4342
# of queries: 45, number of bins in ISOMER: 6906
# of queries: 50, number of bins in ISOMER: 10310
# of queries: 55, number of bins in ISOMER: 15125
# of queries: 60, number of bins in ISOMER: 21262
# of queries: 65, number of bins in ISOMER: 29114
'''


def generate_dataset(size=100000, seed=1):
    np.random.seed(seed)
    cov = np.array([[0.09, 0.04], [0.04, 0.09]])
    mean = np.array([0.5, 0.5])
    return np.random.multivariate_normal(mean, cov, size)


def count_tuple(data, boundary):
    assert (len(boundary) == 4)
    D = data
    D = D[np.logical_and(D[:, 0] >= boundary[0], D[:, 0] <= boundary[2])]
    D = D[np.logical_and(D[:, 1] >= boundary[1], D[:, 1] <= boundary[3])]
    return D.shape[0]


qrid = [0]


def generate_random_queries(data, size=10, seed=0):
    np.random.seed(seed)
    boundaries = np.random.rand(size, 4)
    boundaries[:, 0:2] = boundaries[:, 0:2] * 0.5
    boundaries[:, 2:4] = boundaries[:, 2:4] * 0.5 + 0.5
    N = float(data.shape[0])
    freqs = map(lambda b: count_tuple(data, b) / N, boundaries)

    queries = []
    for i in range(size):
        # print boundaries[i]
        queries.append(Query(boundaries[i], freqs[i], qrid[0]))
        qrid[0] = qrid[0] + 1
    return queries


def generate_permanent_queries(data):
    nx = 10
    ny = 10
    xstep = 1.0 / nx
    ystep = 1.0 / ny
    boundaries = []
    # np.zeros((nx + ny - 1, 4))     # last constraint is redundant
    for i in range(nx - 1):
        xstart = xstep * i
        xend = xstep * (i + 1)
        boundaries.append([xstart, 0.0, xend, 1.0])
    for i in range(ny - 1):
        ystart = ystep * i
        yend = ystep * (i + 1)
        boundaries.append([0.0, ystart, 1.0, yend])
    xstart = xstep * (nx - 1)
    xend = xstep * nx
    boundaries.append([xstart, 0.0, xend, 1.0])
    N = float(data.shape[0])
    freqs = map(lambda b: count_tuple(data, b) / N, boundaries)

    queries = []
    for i in range(len(boundaries)):
        queries.append(Query(boundaries[i], freqs[i], qrid[0]))
        qrid[0] = qrid[0] + 1
    return queries


def generate_test_queries(data, size=(30, 30)):
    nx = size[0]
    ny = size[1]
    xstep = 1.0 / nx
    ystep = 1.0 / ny
    e = 1e-6
    N = float(data.shape[0])

    queries = []
    freqmat = np.zeros((ny, nx))
    for i in range(nx):
        for j in range(ny):
            xstart = xstep * i
            xend = xstep * (i + 1) - e
            ystart = ystep * j
            yend = ystep * (j + 1) - e
            b = (xstart, ystart, xend, yend)
            c = count_tuple(data, b)
            queries.append(Query(b, c / N, None))
            freqmat[j, i] = c
    return queries, freqmat


def viz_freqmap(freqmap):
    # plt.imshow(freqmap, cmap='hot', interpolation='nearest')
    plt.imshow(freqmap, cmap='hot', interpolation='nearest', vmin=0, vmax=0.0025)


def build_isomer(past_queries):
    root = Node(Query([0, 0, 1, 1], 0.830340, 0))
    qid = 1  # we are going to renumber the past queries
    for i, q in enumerate(past_queries):
        q.uid = qid
        qid = qid + 1
        # print q.uid, q
        root.crack(Node(q))
        if i > 0 and i % 5 == 0:
            print
            '# of queries: %d, number of bins in ISOMER: %d' % (i, root.count())
    print
    'Number of total bins in ISOMER: %d' % root.count()
    root.assign_optimal_freq()
    return root


def test_isomer(data, past_queries, test_queries):
    root = build_isomer(past_queries)
    # print root
    print
    'total number of regions: %d' % root.count()

    isomer_answers = map(lambda t: root.answer(t), test_queries)
    groundtruth_answers = map(lambda t: t.freq, test_queries)

    a = np.array(isomer_answers)
    a.shape = (30, 30)
    plt.subplot(1, 3, 1)
    viz_freqmap(a)
    g = np.array(groundtruth_answers)
    g.shape = (30, 30)
    plt.subplot(1, 3, 2)
    viz_freqmap(g)
    plt.subplot(1, 3, 3)
    viz_freqmap(np.abs(g - a))

    print
    np.linalg.norm(a - g)
    print
    'sum of estimates: %f' % np.sum(a)
    print
    'sum of groundtruth: %f' % np.sum(g)

    plt.show()


def test_crumbs(data, past_queries, test_queries):
    quickSel = Crumbs()
    quickSel.assign_optimal_freq(past_queries)

    crumbs_answers = map(lambda t: quickSel.answer(t), test_queries)
    groundtruth_answers = map(lambda t: t.freq, test_queries)

    a = np.array(crumbs_answers)
    a.shape = (30, 30)
    plt.subplot(1, 3, 1)
    viz_freqmap(a)
    g = np.array(groundtruth_answers)
    g.shape = (30, 30)
    plt.subplot(1, 3, 2)
    viz_freqmap(g)
    plt.subplot(1, 3, 3)
    viz_freqmap(np.abs(g - a))

    print
    np.linalg.norm(a - g)
    print
    'sum of estimates: %f' % np.sum(a)
    print
    'sum of groundtruth: %f' % np.sum(g)

    plt.show()


def test_both(data, past_queries, test_queries):
    root = build_isomer(perma_queries[:18] + past_queries[:5])
    isomer_answers = map(lambda t: root.answer(t), test_queries)
    quickSel = Crumbs()
    A, b, v = quickSel.assign_optimal_freq(past_queries + perma_queries)
    # A, b, v = quickSel.assign_optimal_freq(perma_queries)
    crumbs_answers = map(lambda t: quickSel.answer(t), test_queries)
    groundtruth_answers = map(lambda t: t.freq, test_queries)

    plt.subplot(1, 3, 1)
    a = np.array(isomer_answers)
    a.shape = (30, 30)
    viz_freqmap(a)
    print
    'sum of isomer\' estimates: %f' % np.sum(a)

    plt.subplot(1, 3, 2)
    a = np.array(crumbs_answers)
    a.shape = (30, 30)
    viz_freqmap(a)
    print
    'sum of quickSel\' estimates: %f' % np.sum(a)

    plt.subplot(1, 3, 3)
    g = np.array(groundtruth_answers)
    g.shape = (30, 30)
    viz_freqmap(g)
    print
    'sum of groundtruth: %f' % np.sum(g)
    plt.show(block=True)


data = generate_dataset()
past_queries = generate_random_queries(data, size=20)
perma_queries = generate_permanent_queries(data)
test_queries, freqmat = generate_test_queries(data)

import matplotlib.pyplot as plt

# test_isomer(data, perma_queries + past_queries, test_queries)
# test_crumbs(data, past_queries + perma_queries, test_queries)

test_both(data, past_queries, test_queries)

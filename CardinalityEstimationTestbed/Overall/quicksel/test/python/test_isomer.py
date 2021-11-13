'''
node count after 10 queries: 20
node count after 20 queries: 153
node count after 30 queries: 507
node count after 40 queries: 1105
node count after 50 queries: 1922
node count after 60 queries: 2897
node count after 70 queries: 4246
node count after 80 queries: 5815
node count after 90 queries: 7729
node count after 100 queries: 9744
node count after 110 queries: 11972
node count after 120 queries: 14113
node count after 130 queries: 16916
node count after 140 queries: 19943
node count after 150 queries: 23032
node count after 160 queries: 26193
node count after 170 queries: 29787
node count after 180 queries: 33402
node count after 190 queries: 36639
node count after 200 queries: 40613
node count after 210 queries: 44443
total number of nodes after 218 queries: 47664
'''
import sys

sys.path.append('../../src/python')
from quickSel import *
import numpy as np


def test_nodes1():
    n1 = IsomerNode(Hyperrectangle([[0, 2], [0, 2]]))
    n2 = IsomerNode(Hyperrectangle([[1, 3], [1, 3]]))

    print
    n1.total_vol()

    mypiece, otherpieces = n2.crack(n1)
    print
    'my piece:', mypiece
    for o in otherpieces:
        print
        'other piece', o


def test_nodes2():
    n1 = IsomerNode(Hyperrectangle([[0, 4], [0, 4]]))
    n2 = IsomerNode(Hyperrectangle([[1, 2], [1, 2]]))

    mypiece, otherpieces = n2.crack(n1)
    print
    'my piece:', mypiece
    for o in otherpieces:
        print
        'other piece', o


def test_nodes3():
    # new query is larger than an existing one
    n1 = IsomerNode(Hyperrectangle([[1, 2], [1, 2]]))
    n2 = IsomerNode(Hyperrectangle([[0, 4], [0, 4]]))

    mypiece, otherpieces = n2.crack(n1)
    print
    'my piece:', mypiece
    for o in otherpieces:
        print
        'other piece', o


def test_nodes4():
    # new query does not overlap with an existing one
    n1 = IsomerNode(Hyperrectangle([[4, 5], [4, 5]]))
    n2 = IsomerNode(Hyperrectangle([[0, 4], [0, 4]]))

    mypiece, otherpieces = n2.crack(n1)
    print
    'my piece:', mypiece
    for o in otherpieces:
        print
        'other piece', o


def test_nodes5():
    n1 = IsomerNode(Hyperrectangle([[0, 3], [0, 3]]))
    n2 = IsomerNode(Hyperrectangle([[1, 2], [1, 2]]))
    n1.children.append(n2)

    print
    n1.total_vol(), n1.exclusive_vol()

    n3 = IsomerNode(Hyperrectangle([[1.5, 4], [1.5, 4]]))

    mypiece, otherpieces = n3.crack(n1)
    print
    'my piece:', mypiece
    for o in otherpieces:
        print
        'other piece', o


def test_learning1():
    isomer = Isomer(Hyperrectangle([[0, 1], [0, 1]]))
    a1 = Assertion({0: [0, 0.5], 1: [0, 0.5]}, 0.5)
    isomer.assign_optimal_freq([a1])


def test_toy():
    isomer = Isomer(Hyperrectangle([[0, 2], [0, 2]]))
    # a0 = Assertion({0:[0, 2], 1:[0, 2]}, 1.0)
    a1 = Assertion({0: [1, 2], 1: [0, 2]}, 0.8)
    a2 = Assertion({0: [0, 2], 1: [0, 1]}, 0.3)
    isomer.assign_optimal_freq([a1, a2])

    print
    isomer.answer({0: [0, 1], 1: [0, 1]})
    print
    isomer.answer({0: [1, 2], 1: [0, 1]})
    print
    isomer.answer({0: [0, 1], 1: [1, 2]})
    print
    isomer.answer({0: [1, 2], 1: [1, 2]})

    print
    isomer.answer({0: [0, 2], 1: [0, 1]})
    print
    isomer.answer({0: [0, 2], 1: [1, 2]})
    print
    isomer.answer({0: [0, 1], 1: [0, 2]})
    print
    isomer.answer({0: [1, 2], 1: [0, 2]})


def query2DfromBoundary(boundary):
    return {0: (boundary[0], boundary[2]), 1: (boundary[1], boundary[3])}


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


def generate_random_assertions(data, size=10, seed=0):
    np.random.seed(seed)
    boundaries = np.random.rand(size, 4)
    boundaries[:, 0:2] = boundaries[:, 0:2] * 0.5
    boundaries[:, 2:4] = boundaries[:, 2:4] * 0.5 + 0.5
    N = float(data.shape[0])
    freqs = map(lambda b: count_tuple(data, b) / N, boundaries)

    assertions = []
    for i in range(size):
        assertions.append(Assertion({0: (boundaries[i, 0], boundaries[i, 2]),
                                     1: (boundaries[i, 1], boundaries[i, 3])},
                                    freqs[i]))

    return assertions


def generate_permanent_assertions(data):
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
    N = float(data.shape[0])
    freqs = map(lambda b: float(count_tuple(data, b)) / N, boundaries)

    assertions = []
    for i in range(len(boundaries)):
        assertions.append(Assertion({0: (boundaries[i][0], boundaries[i][2]),
                                     1: (boundaries[i][1], boundaries[i][3])},
                                    freqs[i]))
    return assertions


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
            queries.append({0: (b[0], b[2]), 1: (b[1], b[3])})
            freqmat[j, i] = float(c) / N
    return queries, freqmat


import matplotlib.pyplot as plt


def viz_freqmap(freqmap):
    plt.imshow(freqmap, cmap='hot', interpolation='nearest', vmin=0)
    plt.colorbar()


def test_gaussian():
    data = generate_dataset()
    past_assertions = generate_random_assertions(data, size=50)
    perma_assertions = generate_permanent_assertions(data)
    test_queries, freqmat = generate_test_queries(data)

    min_max = Hyperrectangle([[0, 1], [0, 1]])
    isomer = Isomer(min_max)
    total_freq = count_tuple(data, [0, 0, 1, 1]) / float(data.shape[0])
    isomer.assign_optimal_freq(perma_assertions + past_assertions, total_freq)

    # print isomer.root

    answers = map(lambda t: isomer.answer(t), test_queries)

    # for k in quickSel.kernels:
    #    print k

    # print x
    # print np.dot(A, x)
    # print b

    a = np.array(answers)
    a.shape = (30, 30)
    viz_freqmap(a)
    plt.show()


# test_toy()
test_gaussian()

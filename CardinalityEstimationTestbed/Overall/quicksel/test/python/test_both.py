import sys

sys.path.append('../../src/python')
from quickSel import *
import numpy as np


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


def generate_crumbs_permanent_assertions(data):
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


def viz_freqmap(freqmap, ax):
    im = ax.imshow(freqmap, cmap='hot', interpolation='nearest', vmin=0, vmax=0.0025)
    # plt.colorbar(im, fraction=0.046, pad=0.04)


def test_gaussian():
    data = generate_dataset()
    past_assertions = generate_random_assertions(data, size=1000)
    perma_assertions = generate_permanent_assertions(data)
    test_queries, freqmat = generate_test_queries(data)

    ax = plt.subplot(1, 3, 1)
    min_max = Hyperrectangle([[0, 1], [0, 1]])
    isomer = Isomer(min_max)
    total_freq = count_tuple(data, [0, 0, 1, 1]) / float(data.shape[0])
    isomer.assign_optimal_freq(perma_assertions + past_assertions[:500], total_freq)
    answers = map(lambda t: isomer.answer(t), test_queries)
    a = np.array(answers)
    a.shape = (30, 30)
    viz_freqmap(a, ax)
    ax.set_title("ISOMER")

    ax = plt.subplot(1, 3, 2)
    perma_assertions = generate_crumbs_permanent_assertions(data)
    quickSel = Crumbs(min_max)
    quickSel.assign_optimal_freq(perma_assertions + past_assertions[:500])
    answers = map(lambda t: quickSel.answer(t), test_queries)
    a = np.array(answers)
    a.shape = (30, 30)
    viz_freqmap(a, ax)
    ax.set_title("Ours")

    ax = plt.subplot(1, 3, 3)
    viz_freqmap(freqmat, ax)
    ax.set_title("groundtruth")

    # plt.show()
    plt.savefig('comp500.png')


# test_toy()
test_gaussian()

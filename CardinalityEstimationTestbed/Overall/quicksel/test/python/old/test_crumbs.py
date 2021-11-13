import numpy as np
from test_include import *


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
    plt.imshow(freqmap, cmap='hot', interpolation='nearest')


data = generate_dataset()

past_queries = []
past_queries.append(Query([0.2, 0.2, 0.6, 0.6], 0.4, 1))
test_queries, freqmat = generate_test_queries(data)

quickSel = Crumbs()
quickSel.assign_optimal_freq(past_queries)
crumbs_answers = map(lambda t: quickSel.answer(t), test_queries)

import matplotlib.pyplot as plt

a = np.array(crumbs_answers)
a.shape = (30, 30)
viz_freqmap(a)
plt.show()

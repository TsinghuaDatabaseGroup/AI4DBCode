import sys

sys.path.append('../../src/python')
from quickSel import *
import numpy as np


def query2DfromBoundary(boundary):
    return {0: (boundary[0], boundary[2]), 1: (boundary[1], boundary[3])}


def generate_dataset(size=1000, seed=0):
    np.random.seed(seed)
    dataset = np.random.rand(size, 2) * np.array([0.01, 0.8]) + np.array([0.1, 0.1])
    dataset = np.vstack((dataset, np.random.rand(size, 2) * np.array([0.7, 0.01]) + np.array([0.1, 0.9])))
    dataset = np.vstack((dataset, np.random.rand(size, 2) * np.array([0.8, 0.01]) + np.array([0.1, 0.1])))
    dataset = np.vstack((dataset, np.random.rand(size, 2) * np.array([0.7, 0.01]) + np.array([0.2, 0.7])))
    dataset = np.vstack((dataset, np.random.rand(size, 2) * np.array([0.01, 0.6]) + np.array([0.9, 0.1])))
    return dataset


def count_tuple(data, boundary):
    assert (len(boundary) == 4)
    D = data
    D = D[np.logical_and(D[:, 0] >= boundary[0], D[:, 0] <= boundary[2])]
    D = D[np.logical_and(D[:, 1] >= boundary[1], D[:, 1] <= boundary[3])]
    return D.shape[0]


def generate_test_queries(data, size=(100, 100)):
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


def generate_random_assertions(data, size=10, seed=0):
    np.random.seed(seed)

    bottomleft = np.random.rand(size, 2) * np.array([1.5, 1.5]) - np.array([0.5, 0.5])
    boundaries = np.hstack((bottomleft, bottomleft + np.array([0.5, 0.5])))
    # bottomleft = np.random.rand(size,2) * np.array([0.9, 0.9]) + np.array([0.05, 0.05])
    # boundaries = np.hstack( (bottomleft, bottomleft + np.array([0.1, 0.1])) )

    N = float(data.shape[0])
    freqs = map(lambda b: count_tuple(data, b) / N, boundaries)

    assertions = []
    for i in range(size):
        assertions.append(Assertion({0: (boundaries[i, 0], boundaries[i, 2]),
                                     1: (boundaries[i, 1], boundaries[i, 3])},
                                    freqs[i]))
    return assertions


def viz_freqmap(freqmap, ax):
    im = ax.imshow(freqmap, cmap='Blues', interpolation='nearest', vmin=0, vmax=0.006)
    # plt.colorbar(im)


# TEST METHODS

def compile_check():
    a1 = Assertion(query2DfromBoundary([0.1, 0.1, 0.4, 0.4]), 0.5)
    min_max = Hyperrectangle([[0, 1], [0, 1]])
    quickSel = Crumbs(min_max)
    quickSel.assign_optimal_freq([a1])


data = generate_dataset()
past_assertions = generate_random_assertions(data, size=2000)
test_queries, freqmat = generate_test_queries(data)

min_max = Hyperrectangle([[0, 1], [0, 1]])
quickSel = Crumbs(min_max)
A, b, x = quickSel.assign_optimal_freq(past_assertions, k=1)
answers = map(lambda t: quickSel.answer(t), test_queries)

# for k in quickSel.kernels:
#    print k

# print x
# print np.dot(A, x)
# print b


import matplotlib.pyplot as plt

ax = plt.subplot(1, 2, 1)

a = np.array(answers)
a.shape = (100, 100)
viz_freqmap(a, ax)
ax.set_title("estimation")

ax = plt.subplot(1, 2, 2)
viz_freqmap(freqmat.T, ax)
ax.set_title("groundtruth")

plt.show()

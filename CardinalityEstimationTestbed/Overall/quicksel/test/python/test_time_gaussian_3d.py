import sys

sys.path.append('../../src/python')
from quickSel import *

query_dir = '../../test/python/queries/'

data_file_name = query_dir + 'gaussian_3d.data'
query_file_name = query_dir + 'gaussian_trans.query'


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
    # testcases = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # testcases = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
    testcases = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    test_queries = [x[0] for x in queries[1000:2000]]
    test_truth = [x[1] for x in queries[1000:2000]]
    perma_assertions = generate_permanent_assertions(data)
    for num_queries in testcases:
        print
        "Testing Crumbs with " + str(num_queries) + " new queries"
        new_assertions = [Assertion(query[0], query[1]) for query in queries[0:num_queries]]
        min_max = Hyperrectangle([[0, 1], [0, 1], [0, 1]])
        start = time.time()
        quickSel = Crumbs(min_max)
        A, b, x = quickSel.assign_optimal_freq(perma_assertions + new_assertions)
        end = time.time()
        answers = map(lambda t: quickSel.answer(t), test_queries)
        err = []
        rel_err = []
        for i in range(len(answers)):
            err.append(abs(answers[i] - test_truth[i]))
            # rel_err.append(abs(answers[i] - test_queries[i][1]) / float(test_queries[i][1]))
        # avg_rel_err = sum(rel_err)/len(rel_err)
        avg_err = sum(err) / len(err)
        print
        "Avg err is " + str(avg_err)
        print
        "Time used: " + str(end - start)
        print
        "\n\n"


# test isomer
def test_isomer(data, queries):
    testcases = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
    # testcases = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    # testcases = [30]
    test_queries = [x[0] for x in queries[1000:2000]]
    test_truth = [x[1] for x in queries[1000:2000]]
    perma_assertions = generate_permanent_assertions(data)
    for num_queries in testcases:
        print
        "Testing Crumbs with " + str(num_queries) + " new queries"
        new_assertions = [Assertion(query[0], query[1]) for query in queries[0:num_queries]]
        min_max = Hyperrectangle([[0, 1], [0, 1], [0, 1]])
        start = time.time()
        isomer = Isomer(min_max)
        total_freq = count_tuple(data, [0, 0, 0, 1, 1, 1]) / float(len(data))
        isomer.assign_optimal_freq(perma_assertions + new_assertions)
        end = time.time()
        print
        "Time used: " + str(end - start)
        # answers = map(lambda t: isomer.answer(t), test_queries)
        # err = []
        # rel_err = []
        # for i in range(len(answers)):
        #     err.append(abs(answers[i] - test_truth[i]))
        #     # rel_err.append(abs(answers[i] - test_queries[i][1]) / float(test_queries[i][1]))
        # # avg_rel_err = sum(rel_err)/len(rel_err)
        # avg_err = sum(err) / len(err)
        # print "Avg err is " + str(avg_err)
        print
        "\n\n"
        sys.stdout.flush()


data = get_data()
queries = get_queries()
test_crumbs(data, queries)
# test_isomer(data, queries)

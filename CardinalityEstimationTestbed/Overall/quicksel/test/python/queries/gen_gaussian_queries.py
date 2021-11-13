import numpy as np

delta = 0.1;
num_query = 3500;


def dominate(pt1, pt2):
    return pt1[0] >= pt2[0] and pt1[1] >= pt2[1] and pt1[2] >= pt2[2]


def in_range(lower_left, upper_right, point):
    return dominate(upper_right, point) and dominate(point, lower_left)


def check_cold(data, point):
    ll = [point[0] - delta, point[1] - delta, point[2] - delta];
    ur = [point[0] + delta, point[1] + delta, point[2] + delta];
    cnt = sum([in_range(ll, ur, pt) for pt in data])
    if (cnt < len(data) * delta * delta * delta):
        return 1
    return 0


def get_answer(data, ll, ur):
    return sum([in_range(ll, ur, pt) for pt in data])


# cold=0 means all area
# cold=1 means hot area
# cold=2 means cold area
def gen_query(data, cold, size):
    query = []
    freq = []
    while (1):
        [l1, l2, l3] = np.random.uniform(0.05, 0.15, 3);
        center = np.random.uniform(0.05, 0.95, 3);
        if (cold == 1 and check_cold(data, center)):
            continue
        elif (cold == 2 and not check_cold(data, center)):
            continue
        ll = [center[0] - l1 / 2., center[1] - l2 / 2., center[2] - l3 / 2.]
        ur = [center[0] + l1 / 2., center[1] + l2 / 2., center[2] + l3 / 2.]
        ll = [max(0, x) for x in ll]
        ur = [min(1, x) for x in ur]
        query.append([ll, ur])
        freq.append(float(get_answer(data, ll, ur)) / len(data))
        if (len(query) == size):
            break
        print
        len(query)
    return query, freq


def gen_query_trans(data):
    exp_center = [0.49, 0.49, 0.49]
    step = -1.25e-4
    size = 3200
    query = []
    freq = []
    while (1):
        [l1, l2, l3] = np.random.uniform(0.05, 0.15, 3);
        noise = np.random.multivariate_normal([0, 0, 0, ],
                                              [[1e-4, 0, 0],
                                               [0, 1e-4, 0],
                                               [0, 0, 1e-4]])
        center = [exp_center[0] + noise[0], exp_center[1] + noise[1], exp_center[2] + noise[2]]
        ll = [center[0] - l1 / 2., center[1] - l2 / 2., center[2] - l3 / 2.]
        ur = [center[0] + l1 / 2., center[1] + l2 / 2., center[2] + l3 / 2.]
        ll = [max(0, x) for x in ll]
        ur = [min(1, x) for x in ur]
        query.append([ll, ur])
        freq.append(float(get_answer(data, ll, ur)) / len(data))
        if (len(query) == size):
            break
        print
        len(query)
        exp_center = [x + step for x in exp_center]
    return query, freq


def gen_cold(data):
    query, freq = gen_query(data, 2, num_query)
    queryfile = open("gaussian_cold.query", 'w')
    for i in range(0, len(query)):
        ll = query[i][0]
        ur = query[i][1]
        f = freq[i]
        str_ll = ','.join(['%.6f' % x for x in ll])
        str_ur = ','.join(['%.6f' % x for x in ur])
        str_f = '%.7f' % f
        queryfile.write(";".join([str_ll, str_ur, str_f]) + "\n")
    queryfile.close()


def gen_hot(data):
    query, freq = gen_query(data, 1, num_query)
    queryfile = open("gaussian_hot.query", 'w')
    for i in range(0, len(query)):
        ll = query[i][0]
        ur = query[i][1]
        f = freq[i]
        str_ll = ','.join(['%.6f' % x for x in ll])
        str_ur = ','.join(['%.6f' % x for x in ur])
        str_f = '%.7f' % f
        queryfile.write(";".join([str_ll, str_ur, str_f]) + "\n")
    queryfile.close()


def gen_all(data):
    query, freq = gen_query(data, 0, num_query)
    queryfile = open("gaussian_all.query", 'w')
    for i in range(0, len(query)):
        ll = query[i][0]
        ur = query[i][1]
        f = freq[i]
        str_ll = ','.join(['%.6f' % x for x in ll])
        str_ur = ','.join(['%.6f' % x for x in ur])
        str_f = '%.7f' % f
        queryfile.write(";".join([str_ll, str_ur, str_f]) + "\n")
    queryfile.close()


def gen_trans(data):
    query, freq = gen_query_trans(data)
    queryfile = open("gaussian_trans.query", 'w')
    for i in range(len(query)):
        ll = query[i][0]
        ur = query[i][1]
        f = freq[i]
        str_ll = ','.join(['%.6f' % x for x in ll])
        str_ur = ','.join(['%.6f' % x for x in ur])
        str_f = '%.7f' % f
        queryfile.write(";".join([str_ll, str_ur, str_f]) + "\n")
    queryfile.close()


data = []
datafile = open("gaussian_3d.data", 'r')
for line in datafile.readlines():
    data.append([float(x) for x in line.split(',')])
datafile.close()
# gen_cold(data)
# gen_hot(data)
gen_trans(data)
gen_all(data)

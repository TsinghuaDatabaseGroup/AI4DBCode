import numpy as np

num_query = 30;


def dominate(pt1, pt2):
    return pt1[0] >= pt2[0] and pt1[1] >= pt2[1]


def in_range(lower_left, upper_right, point):
    return dominate(upper_right, point) and dominate(point, lower_left)


def get_answer(data, ll, ur):
    return sum([in_range(ll, ur, pt) for pt in data])


def gen_query(data, size):
    query = []
    freq = []
    while (1):
        ll = [np.random.randint(1, 28), np.random.randint(1, 28)]
        ur = [ll[0] + 2, ll[1] + 2]
        query.append([ll, ur])
        freq.append(float(get_answer(data, ll, ur)) / len(data))
        if (len(query) == size):
            break;
        print
        len(query)
    return query, freq


def write_query_to_file(data):
    query, freq = gen_query(data, num_query)
    queryfile = open("zeta_2d.query", 'w')
    for i in range(0, len(query)):
        ll = query[i][0]
        ur = query[i][1]
        f = freq[i]
        str_ll = ','.join([str(x) for x in ll])
        str_ur = ','.join([str(x) for x in ur])
        str_f = '%.3f' % f
        queryfile.write(";".join([str_ll, str_ur, str_f]) + "\n")
    queryfile.close()


data = []
datafile = open("zeta_2d.data", 'r')
for line in datafile.readlines():
    data.append([int(x) for x in line.split(',')])
datafile.close()
write_query_to_file(data)

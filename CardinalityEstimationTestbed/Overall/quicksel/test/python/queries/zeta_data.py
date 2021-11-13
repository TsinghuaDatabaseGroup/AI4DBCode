import numpy as np


def gen_zeta_2d(size=100000, seed=1):
    np.random.seed(seed)
    maxVal = 30
    data = []
    while (1):
        x = np.random.zipf(1.2)
        y = np.random.zipf(1.5)
        if (x > maxVal or y > maxVal):
            continue
        data.append([x, y])
        if (len(data) == size):
            break
    return data


def tostring(point):
    return str(point[0]) + "," + str(point[1]) + "\n"


data = gen_zeta_2d()
datafile = open("zeta_2d.data", 'w')
datafile.truncate()
for point in data:
    datafile.write(tostring(point))
datafile.close()

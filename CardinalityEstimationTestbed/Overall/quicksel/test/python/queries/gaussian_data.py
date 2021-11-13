import numpy as np


def gen_gaussian_3d(size=100000, seed=1):
    np.random.seed(seed)
    mean = [0.5, 0.5, 0.5]
    cov = [[0.1, 0.06, 0.02], [0.06, 0.1, 0.03], [0.02, 0.03, 0.1]]
    return np.random.multivariate_normal(mean, cov, size)


def tostring(point):
    return ("%.3f" % point[0]) + "," + ("%.3f" % point[1]) + "," + ("%.3f" % point[2]) + "\n"


data = gen_gaussian_3d()
datafile = open("gaussian_3d.data", 'w')
datafile.truncate()
for point in data:
    datafile.write(tostring(point))
datafile.close()

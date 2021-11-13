import numpy as np
from quickSel import itr_scaling
from test_include import *


def test1():
    A = np.array([[1, 1, 1, 1],
                  [0, 1, 0, 1],
                  [0, 0, 1, 1.0]])
    b = np.array([1.0, 0.8, 0.3])
    v = np.array([1.0, 1.0, 1.0, 1.0])

    x = itr_scaling.solve(A, b, v)
    print
    x


A = np.array([[1, 1, 1, 1],
              [0, 1, 0, 1],
              [0, 0, 1, 1.0]])
b = np.array([1.0, 0.8, 0.3])
v = np.array([1.0, 1.0, 1.0, 1.0])

e = 1e-6
maxiters = 100
x = itr_scaling.c_iterative_scaling_solver(A, b, v, e, maxiters)
print
x

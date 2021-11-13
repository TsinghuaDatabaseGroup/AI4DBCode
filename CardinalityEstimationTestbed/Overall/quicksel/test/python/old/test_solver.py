from cvxopt import matrix, spmatrix, log, div
from test_include import *

root = None

queries = []
queries.append(gen_query([0, 0, 1, 1]))
queries.append(gen_query([.1, .1, .5, .5]))
queries.append(gen_query([.2, .2, .4, .4]))
queries.append(gen_query([.3, .3, .6, .6]))

for q in queries:
    if root is None:
        root = Node(q)
    else:
        root.crack(Node(q))

n = 4  # number of queries (including the base one)
e, qdict = root.extract()
p = len(e)  # number of variables
print
e

for k, v in qdict.iteritems():
    print
    k, v

import sys

sys.exit(0)

A = np.zeros((n, p))
b = np.zeros((n, 1))
v = np.zeros((p, 1))

for j, a in enumerate(e):
    vol = a[1]
    qn = a[2]
    for q in qn:
        A[q, j] = 1
    v[j] = vol

for i in range(n):
    b[i] = queries[i].freq

b = matrix(b)
v = matrix(v)
logv = log(v)
G = matrix(-1 * np.eye(p))
h = matrix(np.zeros((p, 1)))
Aeq = matrix(A)
beq = matrix(b)

# print Aeq
# print beq
# print G
# print h
# print logv


print
'Convex nonlinear programming (entropy)'


def F(x=None, z=None):
    if x is None: return 0, matrix(1.0 / p, (p, 1))
    if min(x) < 0: return None
    f = x.T * log(x) - x.T * logv
    Df = (log(x) + 1 - logv).T
    if z is None: return f, Df
    H = spmatrix(div(1, x), range(p), range(p))
    return f, Df, H


dims = {'l': p, 'q': [], 's': []}

start_time = time.time()
sol = solvers.cp(F, G, h, dims, Aeq, beq)
x = np.array(sol['x'])
elapsed_time = time.time() - start_time

print
'max ent'
print
x
print

# Q = matrix(np.eye(p))
# q = matrix(np.zeros((p,1)))
#
# start_time = time.time()
# sol = solvers.qp(Q, q, G, h, Aeq, beq)
# x = np.array(sol['x'])
# elapsed_time = time.time() - start_time
#
# print 'qp'
# print x
# print

print
root
print
'total %d distinct regions' % root.count()

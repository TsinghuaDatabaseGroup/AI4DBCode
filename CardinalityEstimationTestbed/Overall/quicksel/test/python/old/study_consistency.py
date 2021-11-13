from test_include import *
import numpy as np

past_queries = []
past_queries.append(Query([0.1, 0, 0.7, 1], 1))
past_queries.append(Query([0.2, 0, 0.3, 1], 0))
past_queries.append(Query([0.3, 0, 0.4, 1], 1))
past_queries.append(Query([0.25, 0.2, 0.37, 0.8], 1))
past_queries.append(Query([0.36, 0.1, 0.45, 0.21], 1))

quickSel = Crumbs()
A, b, x = quickSel.assign_optimal_freq(past_queries)

print
np.dot(A, x)
print
b[:, 0]
print
np.dot(A, x) - b[:, 0]
# print x
#
# print A

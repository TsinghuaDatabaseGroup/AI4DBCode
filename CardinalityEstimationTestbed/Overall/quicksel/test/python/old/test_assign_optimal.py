from test_include import *
import numpy as np
import time
import random

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

node_freq = root.assign_optimal_freq()
# pp.pprint(node_freq)

print
root
print

qid[0] = 0

for n in [15]:
    root = Node(gen_query((0, 0, 1, 1)))
    for i in range(n):
        node1 = Node(gen_query((random.random() * 0.5, random.random() * 0.5,
                                random.random() * 0.5 + 0.5, random.random() * 0.5 + 0.5)))
        print
        node1
        root.crack(node1)

    print
    'total number of regions: %d' % root.count()
    root.assign_optimal_freq()
    # print root

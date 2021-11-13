from test_include import *

root = Node(gen_query([0, 0, 1, 1]))

# inserting a new node that are disjoint each other
node1 = Node(Query([.1, .1, .3, .3], 0.5, 1))
root.crack(node1)
node2 = Node(Query([.5, .5, .7, .7], 0.5, 2))
root.crack(node2)
root.assign_optimal_freq()
print
root, '\n'

query1 = Query([.2, .2, .6, .6], None, None)
print
root.answer(query1)

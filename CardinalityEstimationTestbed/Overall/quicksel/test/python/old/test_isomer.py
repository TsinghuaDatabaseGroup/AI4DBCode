from test_include import *

root = Node(gen_query([0, 0, 1, 1]))

# inserting a new node that are disjoint each other
node1 = Node(gen_query([.1, .1, .3, .3]))
root.crack(node1)
node2 = Node(gen_query([.5, .5, .7, .7]))
root.crack(node2)
print
root, '\n'

# inserting a new node that overlaps only one child
node3 = Node(gen_query([.2, .2, .4, .4]))
root.crack(node3)
print
root, '\n'

# cracking the nodes with children already
qid = 0
root = Node(gen_query([0, 0, 1, 1]))
node1 = Node(gen_query([.1, .1, .5, .5]))
root.crack(node1)
node2 = Node(gen_query([.2, .2, .4, .4]))
root.crack(node2)
node3 = Node(gen_query([.3, .3, .6, .6]))
root.crack(node3)
print
root
print
'total %d number of regions' % root.count()
pp.pprint(root.extract())
print
'\n'

# inserting the same node
qid = 0
root = Node(gen_query([0, 0, 1, 1]))
node1 = Node(gen_query([.1, .1, .5, .5]))
root.crack(node1)
node2 = Node(gen_query([.1, .1, .5, .5]))
root.crack(node2)
print
root
print
'total %d number of regions' % root.count()
pp.pprint(root.extract())

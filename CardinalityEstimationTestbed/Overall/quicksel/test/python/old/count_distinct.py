'''
#quereis #bins
5 26
10 779
20 7733
'''
import imp

isomer = imp.load_source('isomer', '../../src/python/isomer.py')

import random


def Node(query):
    return isomer.STHoles2d(query.boundary, [query])


def Query(boundary, freq, uid):
    return isomer.Query2d(boundary, freq, uid)


qid = 0


def gen_query(boundary):
    boundary = isomer.Boundary(boundary)
    freq = (boundary.r - boundary.l) * (boundary.t - boundary.b)
    global qid
    query = Query(boundary, freq, qid)
    qid = qid + 1
    return query


for n in [5]:
    root = Node(gen_query((0, 0, 1, 1)))
    for i in range(n):
        node1 = Node(gen_query((random.random() * 0.5, random.random() * 0.5,
                                random.random() * 0.5 + 0.5, random.random() * 0.5 + 0.5)))
        print
        node1
        root.crack(node1)

    print
    print
    root
    print
    root.count()

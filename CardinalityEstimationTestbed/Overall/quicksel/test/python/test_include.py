import sys

sys.path.append('../../src/python')
# isomer = imp.load_source('isomer', '../../src/python/isomer.py')
# import imp
import pprint
from quickSel import isomer

pp = pprint.PrettyPrinter(indent=2)


def Node(query):
    return isomer.STHoles2d(query.boundary, [query])


def Query(boundary, freq, uid=None):
    return isomer.Query2d(boundary, freq, uid)


def Boundary(boundary):
    return isomer.Boundary(boundary)


qid = [0]


def gen_query(boundary):
    boundary = isomer.Boundary(boundary)
    freq = (boundary.r - boundary.l) * (boundary.t - boundary.b)
    query = Query(boundary, freq, qid[0])
    qid[0] = qid[0] + 1
    return query

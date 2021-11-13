import math
import subprocess

username = 'postgres'
database = 'postgres'
order_cardinality = 100
cart_cardinality = 145
total = 32434489


def generate_hinted_query(l1, u1, l2, u2, sel):
    order_up = int(math.ceil(u1 * order_cardinality))
    order_lo = int(math.floor(l1 * order_cardinality))
    cart_up = int(math.ceil(u2 * cart_cardinality))
    cart_lo = int(math.floor(l2 * cart_cardinality))
    rows = int(sel * total)
    if rows == 0:
        rows = 1
    return "psql -c 'SET max_parallel_workers_per_gather = 0' -c 'LOAD \'\\\'pg_hint_plan\\' -c '\\timing' -c '/*+ Leading((p (op o))) Rows(o op #{})*/ select count(*) " \
           "from (select op.product_id as pid from orders o JOIN order_products op " \
           "ON o.order_id=op.order_id " \
           "and o.order_number>={} " \
           "and o.order_number<={} " \
           "and op.add_to_cart>={} " \
           "and op.add_to_cart<={}) tmp JOIN products p ON tmp.pid=p.product_id'".format(rows, order_lo, order_up,
                                                                                         cart_lo, cart_up)


def generate_query(l1, u1, l2, u2):
    order_up = int(math.ceil(u1 * order_cardinality))
    order_lo = int(math.floor(l1 * order_cardinality))
    cart_up = int(math.ceil(u2 * cart_cardinality))
    cart_lo = int(math.floor(l2 * cart_cardinality))
    return "psql -c 'SET max_parallel_workers_per_gather = 0' -c 'LOAD  \'\\\'pg_hint_plan\\' -c '\\timing' -c '/*+ Leading((p (op o))) */ select count(*) " \
           "from (select op.product_id as pid from orders o JOIN order_products op " \
           "ON o.order_id=op.order_id " \
           "and o.order_number>={} " \
           "and o.order_number<={} " \
           "and op.add_to_cart>={} " \
           "and op.add_to_cart<={}) tmp JOIN products p ON tmp.pid=p.product_id'".format(order_lo, order_up,
                                                                                         cart_lo, cart_up)


def hinted_queries(sel_file, query_file):
    print("Testing QuickSel")
    qf = open(query_file, 'r')
    sf = open(sel_file, 'r')

    # discard first 50 lines
    for i in range(50):
        qf.readline()

    while True:
        line1 = qf.readline()
        line2 = sf.readline()
        if not line2:
            break
        l1 = float(line1.split(',')[0])
        u1 = float(line1.split(',')[1])
        l2 = float(line1.split(',')[2])
        u2 = float(line1.split(',')[3])
        sel = float(line2)
        query = generate_hinted_query(l1, u1, l2, u2, sel)
        output = \
            subprocess.Popen([query], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True).communicate()[
                0].decode('utf-8')
        # print(output)
        timing = float(output.split('\n')[8].split(' ')[1])
        print(timing, flush=True)
    qf.close()
    sf.close()


def psql_queries(sel_file, query_file):
    print("Testing PostgreSQL")
    qf = open(query_file, 'r')
    sf = open(sel_file, 'r')

    # discard first 50 lines
    for i in range(50):
        qf.readline()

    while True:
        line1 = qf.readline()
        line2 = sf.readline()
        if not line2:
            break
        l1 = float(line1.split(',')[0])
        u1 = float(line1.split(',')[1])
        l2 = float(line1.split(',')[2])
        u2 = float(line1.split(',')[3])
        query = generate_query(l1, u1, l2, u2)
        output = \
            subprocess.Popen([query], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True).communicate()[
                0].decode('utf-8')
        # print(output)
        timing = float(output.split('\n')[8].split(' ')[1])
        print(timing, flush=True)
    qf.close()
    sf.close()


hinted_queries('/home/ubuntu/experiment/sigmod_rev_instacart_correlated_sel_isomer_withPermanent.txt',
               '/home/ubuntu/experiment/sigmod20_rev_correlated_instacart_queries.csv')
# psql_queries('/home/ubuntu/experiment/sigmod_rev_instacart_correlated_sel_quicksel.txt', '/home/ubuntu/experiment/sigmod20_rev_correlated_instacart_queries.csv')

'''
test=# select min(l_shipdate), max(l_shipdate) from lineitem;
    min     |    max     
------------+------------
 1992-01-02 | 1998-12-01
(1 row)
'''
import time

import psycopg2

conn = psycopg2.connect("dbname=test user=postgres")
cur = conn.cursor()
cur.execute('set search_path to tpch50g')
cur.execute("set work_mem to '8GB'")
cur.execute('set enable_hashjoin to on')
cur.execute('set enable_mergejoin to off')
cur.execute('set enable_nestloop to off')

prices = list(range(0, 550000, 50000))
prices.extend(range(510000, 580000, 10000))


def get_counts():
    for price in prices:
        sql = '''
        select count(*)
        from orders
        where o_totalprice >= %d
        ''' % (price)

        cur.execute(sql)
        c = cur.fetchone()[0]
        print('price >= %d, count: %d ' % (price, c))


def run_all_queries():
    for price in prices:
        sql = '''
        select avg(l_quantity)
        from lineitem inner join orders
          on l_orderkey = o_orderkey
        where o_totalprice >= %d
        ''' % (price)

        start_time = time.time()
        cur.execute(sql)
        cur.fetchall()
        elapsed_time = time.time() - start_time
        print('price >= %d, elapsed time: %.6f sec' % (price, elapsed_time))


get_counts()
# cur.execute('set enable_indexscan to on')

# default
print('default scan mode')
for i in range(5):
    run_all_queries()

# sequential
print('seq scan only')
for i in range(5):
    cur.execute('set enable_seqscan to on')
    cur.execute('set enable_bitmapscan to off')
    cur.execute('set enable_indexscan to off')
    run_all_queries()

# index
print('index scan only')
for i in range(5):
    cur.execute('set enable_seqscan to off')
    cur.execute('set enable_bitmapscan to on')
    cur.execute('set enable_indexscan to on')
    run_all_queries()

cur.close()
conn.close()

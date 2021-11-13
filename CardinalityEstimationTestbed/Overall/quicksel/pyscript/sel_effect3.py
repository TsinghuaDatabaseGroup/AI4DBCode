'''
test=# select min(l_extendedprice), max(l_extendedprice) from lineitem;
  min   |    max    
--------+-----------
 900.51 | 104949.50
(1 row)
'''
import time

import psycopg2

conn = psycopg2.connect("dbname=test user=postgres")
cur = conn.cursor()
cur.execute('set search_path to tpch10g')
cur.execute("set work_mem to '2GB'")

# prices = list(range(10000, 100001, 10000))
prices = list(range(60000, 70001, 1000))


def get_counts():
    for price in prices:
        sql = '''
        select count(*)
        from lineitem
        where l_extendedprice >= %d
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
        where l_extendedprice >= %d
          and o_totalprice >= 100;
        ''' % (price)

        start_time = time.time()
        cur.execute(sql)
        cur.fetchall()
        elapsed_time = time.time() - start_time
        print('price >= %d, elapsed time: %.6f sec' % (price, elapsed_time))


def turn_on_seq_only():
    print('seq')
    cur.execute('set enable_seqscan to on')
    cur.execute('set enable_indexscan to off')
    cur.execute('set enable_bitmapscan to off')


def turn_on_index_scan_only():
    print('index scan')
    cur.execute('set enable_seqscan to off')
    cur.execute('set enable_indexscan to on')
    cur.execute('set enable_bitmapscan to off')


get_counts()

## default
# print('default scan mode')
# for i in range(5):
#    run_all_queries()

# merge join only
print('merge join only')
cur.execute('set enable_hashjoin to off')
cur.execute('set enable_mergejoin to on')
cur.execute('set enable_nestloop to off')

turn_on_seq_only()
run_all_queries()
print()

turn_on_index_scan_only()
run_all_queries()
print()

# hash join only
print('hash join only')
cur.execute('set enable_hashjoin to on')
cur.execute('set enable_mergejoin to off')
cur.execute('set enable_nestloop to off')

turn_on_seq_only()
run_all_queries()
print()

turn_on_index_scan_only()
run_all_queries()
print()

# nested loop join only
print('nested loop join only')
cur.execute('set enable_hashjoin to off')
cur.execute('set enable_mergejoin to off')
cur.execute('set enable_nestloop to on')

turn_on_index_scan_only()
run_all_queries()
print()

cur.close()
conn.close()

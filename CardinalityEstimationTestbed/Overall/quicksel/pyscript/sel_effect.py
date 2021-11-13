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

start_dates = [
    '1992-01-01', '1993-01-01', '1994-01-01', '1995-01-01', '1996-01-01', '1997-01-01',
    '1998-01-01', '1998-02-01', '1998-03-01', '1998-04-01', '1998-05-01', '1998-06-01',
    '1998-07-01', '1998-08-01', '1998-08-02'
]


def get_counts():
    for start_date in start_dates:
        sql = '''
        select count(*)
        from orders
        where o_orderdate between '%s' and '1998-08-02'
        ''' % (start_date)

        cur.execute(sql)
        c = cur.fetchone()[0]
        print('start date: %s, count: %d ' % (start_date, c))


def run_all_queries():
    for start_date in start_dates:
        sql = '''
        select avg(l_quantity)
        from lineitem inner join orders
          on l_orderkey = o_orderkey
        where o_orderdate between '%s' and '1998-08-02'
        ''' % (start_date)

        start_time = time.time()
        cur.execute(sql)
        cur.fetchall()
        elapsed_time = time.time() - start_time
        print('start date: %s, elapsed time: %.6f sec' % (start_date, elapsed_time))


# get_counts()
cur.execute('set enable_indexscan to on')

for i in range(5):
    print('hash join')
    cur.execute('set enable_hashjoin to on')
    cur.execute('set enable_mergejoin to off')
    cur.execute('set enable_nestloop to off')
    run_all_queries()

# print('nested loop join')
# cur.execute('set enable_hashjoin to off')
# cur.execute('set enable_mergejoin to off')
# cur.execute('set enable_nestloop to on')
# run_all_queries()

cur.close()
conn.close()

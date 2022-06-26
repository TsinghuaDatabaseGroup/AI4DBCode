import os
import time
import base
import sys


def sqlsmith_generate_queries(dbname, num_queries, target_path):
    command = '''sqlsmith --target=\"host={0} user={1} dbname={2}\" --exclude-catalog --dry-run --max-queries={3} > {4}
    '''.format("localhost", "lixizhang", dbname, num_queries, target_path)
    os.system(command)


def cal_point_time(dbname, pc, error, N, type, log_path, query_to_path):
    print("Generating {} queries meet point {} constraint:{} with acceptable error {} to '{}'".format(N, type, pc,
                                                                                                      error, log_path))
    time.sleep(10)
    satisfied_count = 0
    total_count = 0
    # print("log_path:", log_path)
    if os.path.exists(log_path):
        log = open(log_path, 'r+')
        lines = log.readlines()
        last_line = lines[-1]
        print(last_line)
        satisfied_count = int(last_line.split(';')[1].split(':')[1])
        total_count = int(last_line.split(';')[2].split(':')[1])
    else:
        log = open(log_path, 'w')
    log.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), satisfied_count, total_count))
    low_bound = pc - error
    up_bound = pc + error
    while satisfied_count < N:
        sqlsmith_generate_queries(dbname, N, query_to_path)
        total_count += N
        queries = []
        with open(query_to_path, 'r') as f:
            queries.extend(f.read().split(';'))
            for query in queries:
                try:
                    result, e_info = base.get_evaluate_query_info(dbname, query)
                    if result:
                        if type == "cost":
                            if low_bound <= e_info['total_cost'] <= up_bound:
                                satisfied_count += 1
                                if satisfied_count % 100 == 0:
                                    log.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), satisfied_count, total_count))
                                if total_count % 1000 == 0:
                                    print("{}/{} time: {}\n".format(satisfied_count, total_count, str(time.time())))
                        elif type == "card":
                            if low_bound <= e_info['e_cardinality'] <= up_bound:
                                satisfied_count += 1
                                if satisfied_count % 100 == 0:
                                    log.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), satisfied_count, total_count))
                                if total_count % 1000 == 0:
                                    print("{}/{} time: {}\n".format(satisfied_count, total_count, str(time.time())))
                        else:
                            print("error")
                            return
                except Exception as result:
                    log.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), satisfied_count, total_count))
                    log.close()
                    print(result)
    log.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), satisfied_count, total_count))
    log.close()


def cal_range_time(dbname, rc, N, type, log_path, query_to_path):
    print("Generating {} queries meet range {} constraint:[{}, {}] to '{}'".format(N, type, rc[0], rc[1], log_path))
    time.sleep(10)
    satisfied_count = 0
    total_count = 0

    if os.path.exists(log_path):
        log = open(log_path, 'r+')
        lines = log.readlines()
        last_line = lines[-1]
        print(last_line)
        satisfied_count = int(last_line.split(';')[1].split(':')[1])
        total_count = int(last_line.split(';')[2].split(':')[1])
    else:
        log = open(log_path, 'w')

    log.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), satisfied_count, total_count))
    low_bound = rc[0]
    up_bound = rc[1]

    while satisfied_count < N:
        sqlsmith_generate_queries(dbname, N, query_to_path)
        total_count += N
        queries = []
        with open(query_to_path, 'r') as f:
            queries.extend(f.read().split(';'))
            for query in queries:
                try:
                    result, e_info = base.get_evaluate_query_info(dbname, query)
                    if result:
                        if type == "cost":
                            if low_bound <= e_info['total_cost'] <= up_bound:
                                satisfied_count += 1
                                if satisfied_count % 100 == 0:

                                    log.write(
                                        "time:{};s_count:{};t_count:{}\n".format(str(time.time()), satisfied_count,
                                                                                 total_count))
                                if total_count % 1000 == 0:
                                    print("{}/{} time: {}\n".format(satisfied_count, total_count, str(time.time())))
                        elif type == "card":
                            if low_bound <= e_info['e_cardinality'] <= up_bound:
                                satisfied_count += 1
                                if satisfied_count % 100 == 0:
                                    log.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), satisfied_count,
                                                                                       total_count))
                                if total_count % 1000 == 0:
                                    print("{}/{} time: {}\n".format(satisfied_count, total_count, str(time.time())))
                        else:
                            print("error")
                            return
                except Exception as result:
                    log.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), satisfied_count, total_count))
                    log.close()
                    print(result)
    log.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), satisfied_count, total_count))
    log.close()


# cal_point_time('tpch', pc=0, error=10, N=1000, type='cost')
# cal_point_time('tpch', pc=0, error=10, N=1000, type='card')
# cal_point_time('tpch', pc=1000, error=100, N=1000, type='cost')
# cal_point_time('tpch', pc=1000, error=100, N=1000, type='card')
# cal_point_time('tpch', pc=10000, error=1000, N=1000, type='cost')
# cal_point_time('tpch', pc=10000, error=1000, N=1000, type='card',
#                query_to_path='/home/lixizhang/learnSQL/sqlsmith/tpch/logfile/tmp',
#                log_path='/home/lixizhang/learnSQL/sqlsmith/tpch/logfile/card_pc10000_N1000')

if __name__ == '__main__':
    para = sys.argv
    # print(para)
    dbname = para[1]
    ctype = para[2]         # cost/card
    mtype = para[3]         # point/range
    N = int(para[4])
    cur_path = os.path.abspath('.')
    db_path = cur_path + '/' + dbname + '/logfile'
    if mtype == 'point':
        # print('enter point')
        pc = int(para[5])
        error = pc * 0.1
        log_path = db_path + '/' + '{}_pc{}_N{}'.format(ctype, pc, N)
        tmp_path = db_path + '/' + '{}_pc{}_N{}_tmp'.format(ctype, pc, N)
        cal_point_time(dbname=dbname, type=ctype, N=N, pc=pc, error=error, log_path=log_path, query_to_path=tmp_path)
    elif mtype == 'range':
        rc = (int(para[5]), int(para[6]))
        log_path = db_path + '/' + '{}_rc{}_{}_N{}'.format(ctype, rc[0], rc[1], N)
        tmp_path = db_path + '/' + '{}_rc{}_{}_N{}_tmp'.format(ctype, rc[0], rc[1], N)
        cal_range_time(dbname=dbname, type=ctype, N=N, rc=rc, log_path=log_path, query_to_path=tmp_path)
    else:
        print("error")

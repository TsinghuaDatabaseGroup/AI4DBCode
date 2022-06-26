import psycopg2
import pymysql
from enum import IntEnum
import re
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import sys
SEQ_LENGTH = 40


def connect_server(dbname, server_name='postgresql'):
    if server_name == 'mysql':
        db = pymysql.connect(host="localhost", user="root", passwd="", db=dbname, charset="utf8")
        cursor = db.cursor()
        return db, cursor
    elif server_name == 'postgresql':
        sucess = 0
        db = -1
        cursor = -1
        count = 0
        while not sucess and count < 3:
            try:
                db = psycopg2.connect(database=dbname, user="", password="", host="166.111.5.177", port="5433")
                # db = psycopg2.connect(database=dbname, user="lixizhang", password="xi10261026zhang", host="localhost", port="5432")
                cursor = db.cursor()
                sucess = 1
            except Exception as result:
                count += 1
                time.sleep(10)
        if db == -1 or cursor == -1:
            raise Exception
        return db, cursor

    else:
        print('数据库连接不上...')
        return


class DataType(IntEnum):
    VALUE = 0
    TIME = 1
    CHAR = 2


class Buffer:
    def __init__(self):
        self.states = np.zeros(SEQ_LENGTH, dtype=int)
        self.actions = np.zeros(SEQ_LENGTH, dtype=int)
        self.rewards = np.zeros(SEQ_LENGTH, dtype=float)

    def store(self, state, action, reward, time_step):
        self.states[time_step] = state
        self.actions[time_step] = action
        self.rewards[time_step] = reward

    def clear(self):
        self.states[:] = 0
        self.actions[:] = 0
        self.rewards[:] = 0


AGGREGATE_CONSTRAINTS = {
    DataType.VALUE.value: ['count', 'max', 'min', 'avg', 'sum'],
    DataType.VALUE.CHAR: ['count', 'max', 'min'],
    DataType.VALUE.TIME: ['count', 'max', 'min']
}


def transfer_field_type(database_type, server):
    data_type = list()
    if server == 'mysql':
        data_type = [['int', 'tinyint', 'smallint', 'mediumint', 'bigint', 'float', 'double', 'decimal'],
                     ['date', 'time', 'year', 'datetime', 'timestamp']]
        database_type = database_type.lower().split('(')[0]
    elif server == 'postgresql':
        data_type = [['integer', 'numeric'],
                     ['date']]
    if database_type in data_type[0]:
        return DataType.VALUE.value
    elif database_type in data_type[1]:
        return DataType.TIME.value
    else:
        return DataType.CHAR.value


def build_relation_graph(dbname, schema):
    # 关系图
    print("build relation graph..")
    relation_graph = RelationGraph()
    for table_name in schema.keys():
        relations = get_foreign_relation(dbname, table_name)
        # print(relations)
        for relation in relations:
            relation_graph.add_relation(table_name, relation[2], (relation[1], relation[3]))
            relation_graph.add_relation(relation[2], table_name, (relation[3], relation[1]))
    # print(relation_graph.print_relation_graph())
    print("build relation graph done..")
    return relation_graph


def get_index_key(cursor, table_name, index_name):
    sql = '''select
        t.relname as table_name,
        i.relname as index_name,
        a.attname as column_name
    from
        pg_class t,
        pg_class i,
        pg_index ix,
        pg_attribute a
    where
        t.oid = ix.indrelid
        and i.oid = ix.indexrelid
        and a.attrelid = t.oid
        and a.attnum = ANY(ix.indkey)
        and t.relkind = 'r'
        and t.relname like '{}%'
    order by
        t.relname,
        i.relname;
    '''.format(table_name)
    cursor.execute(sql)
    index_info = cursor.fetchall()
    # print(index_info)
    for info in index_info:
        index_name.append(info[2])


def get_oid(dbname, table_name):
    db, cursor = connect_server(dbname)
    # print(dbname, '-', table_name)
    sql = '''
            SELECT
                c.oid,
                n.nspname,
                c.relname
            FROM pg_catalog.pg_class c
            LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname OPERATOR(pg_catalog.~) '^({})$'
            AND pg_catalog.pg_table_is_visible(c.oid)
            ORDER BY 2, 3;
                    '''.format(table_name)
    cursor.execute(sql)
    data = cursor.fetchall()
    return data[0][2], data[0][0]


def get_foreign_relation(dbname, table_name):
    db, cursor = connect_server(dbname)
    table_name, oid = get_oid(dbname, table_name)
    sql = '''
        SELECT 
                conname,
                pg_catalog.pg_get_constraintdef(r.oid, true) as condef
        FROM pg_catalog.pg_constraint r
        WHERE r.conrelid = '{}' AND r.contype = 'f' ORDER BY 1;
        '''.format(oid)
    cursor.execute(sql)
    data = cursor.fetchall()
    relations = list()
    p1 = re.compile(r'[(](.*?)[)]', re.S)
    for item in data:
        info = item[1].split('REFERENCES')
        to_table = info[1].split('(')[0].strip()
        from_col_info = re.findall(p1, info[0])[0].replace(' ', '').split(',')
        from_col = list()
        for col in from_col_info:
            from_col.append("{}.{}".format(table_name, col))
        to_col_info = re.findall(p1, info[1])[0].replace(' ', '').split(',')
        to_col = list()
        for col in to_col_info:
            to_col.append("{}.{}".format(to_table, col))
        relations.append((table_name, tuple(from_col), to_table, tuple(to_col)))
    return relations


def get_table_structure(dbname, server='postgresql'):
    """
    schema: {table_name: [field_name]}
    :param cursor:
    :return:
    """
    if server == 'mysql':
        db, cursor = connect_server(dbname)
        cursor.execute('SHOW TABLES')
        tables = cursor.fetchall()
        schema = {}
        for table_info in tables:
            table_name = table_info[0]
            sql = 'SHOW COLUMNS FROM ' + table_name
            cursor.execute(sql)
            columns = cursor.fetchall()
            schema[table_name] = {}
            for col in columns:
                schema[table_name][col[0]] = [transfer_field_type(col[1], server), col[3]]
            return schema
    elif server == 'postgresql':
        cur_path = os.path.abspath('.')
        tpath = cur_path + '/sampled_data/'+dbname+'/schema'
        if os.path.exists(tpath):
            with open(tpath, 'r') as f:
                schema = eval(f.read())
        else:
            db, cursor = connect_server(dbname)
            cursor.execute('SELECT table_name FROM information_schema.tables WHERE table_schema = \'public\';')
            tables = cursor.fetchall()
            schema = {}
            for table_info in tables:
                table_name = table_info[0]
                sql = 'SELECT column_name, data_type FROM information_schema.columns WHERE table_name = \'' + table_name + '\';'
                cursor.execute(sql)
                columns = cursor.fetchall()
                schema[table_name] = []
                for col in columns:
                    if transfer_field_type(col[1], server) == DataType.VALUE.value:
                        sql = 'SELECT count({}) FROM {};'.format(col[0], table_name)
                        cursor.execute(sql)
                        num = cursor.fetchall()
                        if num[0][0] != 0:
                            schema[table_name].append(col[0])
            with open(tpath, 'w') as f:
                f.write(str(schema))
            cursor.close()
            db.close()
            #print(schema)
        return schema


def get_table_structure_all(dbname, server='postgresql'):
    """
    schema: {table_name: {field_name {'DataType', 'keytype'}}}
    :param cursor:
    :return:
    """
    if server == 'mysql':
        db, cursor = connect_server(dbname)
        cursor.execute('SHOW TABLES')
        tables = cursor.fetchall()
        schema = {}
        for table_info in tables:
            table_name = table_info[0]
            sql = 'SHOW COLUMNS FROM ' + table_name
            cursor.execute(sql)
            columns = cursor.fetchall()
            schema[table_name] = {}
            for col in columns:
                schema[table_name][col[0]] = [transfer_field_type(col[1], server), col[3]]
        return schema
    elif server == 'postgresql':
        print("get_table_structure_all")
        cur_path = os.path.abspath('.')
        tpath = cur_path + '/sampled_data/' + dbname + '/schema'
        if os.path.exists(tpath):
            with open(tpath, 'r') as f:
                schema = eval(f.read())
        else:
            db, cursor = connect_server(dbname)
            cursor.execute('SELECT table_name FROM information_schema.tables WHERE table_schema = \'public\';')
            tables = cursor.fetchall()
            schema = {}
            for table_info in tables:
                table_name = table_info[0]
                sql = 'SELECT column_name, data_type FROM information_schema.columns WHERE table_name = \'' + table_name + '\';'
                cursor.execute(sql)
                columns = cursor.fetchall()
                schema[table_name] = {}
                for col in columns:
                    sql = 'SELECT count({}) FROM {};'.format(col[0], table_name)
                    cursor.execute(sql)
                    num = cursor.fetchall()
                    if num[0][0] != 0:
                        schema[table_name][col[0]] = [transfer_field_type(col[1], server)]
            with open(tpath, 'w') as f:
                f.write(str(schema))
            cursor.close()
            db.close()
            print(schema)
        return schema


class RelationGraph(object):
    """
    维护表和表外键的关系，这样可以知道哪些表可以连接，有意义的连接
    """
    def __init__(self):
        self.relation_graph = {}

    def add_relation(self, begin, to, relation):
        if begin not in self.relation_graph.keys():
            self.relation_graph[begin] = {}
        if to not in self.relation_graph[begin]:
            self.relation_graph[begin][to] = relation

    def get_relation(self, table):
        return set(self.relation_graph[table].keys())

    def get_relation_key(self, begin, end):
        return self.relation_graph[begin][end]

    def print_relation_graph(self):
        for from_table in self.relation_graph.keys():
            for to_table in self.relation_graph[from_table]:
                print("from:{} to:{}: {}".format(from_table, to_table, self.relation_graph[from_table][to_table]))


def get_evaluate_query_info(dbname, sql):
    conn, cursor = connect_server(dbname)
    try:
        cursor.execute('explain (format json)' + ' ' + sql)
        result = cursor.fetchall()[0][0][0]['Plan']
        query_info = {'e_execute': True,
                      'startup_cost': result['Startup Cost'],
                      'total_cost': result['Total Cost'],
                      'e_cardinality': result['Plan Rows'],
                      }
        cursor.close()
        conn.close()
        return 1, query_info
    except Exception as result:
        cursor.close()
        conn.close()
        return 0, result


def get_execute_query_info(dbname, sql):
    conn, cursor = connect_server(dbname)
    try:
        cursor.execute("set statement_timeout to 60000")
        cursor.execute('explain (analyze, format json)' + ' ' + sql)
        result = cursor.fetchall()[0][0][0]['Plan']
        query_info = {'e_execute': True,
                      # 'r_execute': True,
                      # 'startup_cost': result['Startup Cost'],
                      # 'total_cost': result['Total Cost'],
                      # 'e_cardinality': result['Plan Rows'],
                      'start_time': result['Actual Startup Time'],
                      'total_time': result['Actual Total Time'],
                      'r_cardinality': result['Actual Rows']
                      }
        cursor.close()
        conn.close()
        return 1, query_info
    except Exception as result:
        cursor.close()
        conn.close()
        return 0, result


def cal_file_e_info(dbname, fpath, tpath, cum=False):
    if not os.path.exists(fpath):
        print('文件不存在')
        return
    else:
        queries = []
        with open(fpath, 'r') as f:
            queries.extend(f.read().split(';'))
    if not os.path.exists(tpath) or not cum:
        query_info = pd.DataFrame(columns=['e_execute', 'r_execute', 'startup_cost', 'total_cost', 'e_cardinality',
                                           'start_time', 'total_time', 'r_cardinality',
                                           'remark_execute', 'remark_estimate'])
    else:
        query_info = pd.read_csv(tpath, index_col=0)
    for i in range(len(queries)):
        if queries[i] is '':
            break
        result, e_info = get_evaluate_query_info(dbname, queries[i])
        if i % 100 == 0:
            print('evaluate:', i)
            print(e_info)
        # print(e_info)
        if result:
            query_info = query_info.append([e_info], ignore_index=True)
        else:
            print(e_info)
            info = {
                'e_execute': False,
                'remark_estimate': e_info,
            }
            query_info = query_info.append([info], ignore_index=True)
    query_info.to_csv(tpath)


def cal_file_r_info(dbname, fpath, tpath):
    query_info = pd.read_csv(tpath, index_col=0)
    queries = []
    with open(fpath, 'r') as f:
        queries.extend(f.read().split(';'))
        for i in range(len(queries)):
            if query_info.iloc[i, 0] and pd.isnull(query_info.iloc[i, 1]):
                print('executing ', i)
                result, r_info = get_execute_query_info(dbname, queries[i])
                print('queryid:', i, 'info:', r_info)
                if result:
                    query_info.loc[i, 'r_execute'] = True
                    query_info.loc[i, 'start_time'] = r_info['start_time']
                    query_info.loc[i, 'total_time'] = r_info['total_time']
                    query_info.loc[i, 'r_cardinality'] = r_info['r_cardinality']
                else:
                    query_info.loc[i, 'r_execute'] = False
                    query_info.loc[i, 'remark_execute'] = r_info
                query_info.to_csv(tpath)


def cal_gap(stamp1, stamp2, hours=False):
    t1 = time.localtime(stamp1)
    t2 = time.localtime(stamp2)
    t1 = time.strftime("%Y-%m-%d %H:%M:%S",t1)
    t2 = time.strftime("%Y-%m-%d %H:%M:%S", t2)
    time1 = datetime.datetime.strptime(t1,"%Y-%m-%d %H:%M:%S")
    time2 = datetime.datetime.strptime(t2, "%Y-%m-%d %H:%M:%S")
    gap = time2-time1
    print(gap)
    if hours:
        return gap.total_seconds() / 3600
    return gap



def cal_time(path):
    if not os.path.exists(path):
        print('no file')
        return
    total_time = datetime.timedelta(0)
    with open(path, 'r') as f:
        log = f.readlines()
    # print(log)
    for index in range(1, len(log)):
        cur = log[index]
        before = log[index-1]
        stamp1 = float(before.split(';')[0].split(':')[1])
        stamp2 = float(cur.split(';')[0].split(':')[1])
        s_count1 = int(before.split(';')[1].split(':')[1])
        s_count2 = int(cur.split(';')[1].split(':')[1])
        if s_count1 != s_count2:
            total_time += cal_gap(stamp1, stamp2)
    return total_time


def cal_time_v2(path):
    if not os.path.exists(path):
        print('no file')
        return
    total_time = datetime.timedelta(0)
    with open(path, 'r') as f:
        log = f.readlines()
    # print(log)
    for index in range(1, len(log)):
        cur = log[index]
        before = log[index-1]
        stamp1 = float(before.split(';')[0].split(':')[1])
        stamp2 = float(cur.split(';')[0].split(':')[1])
        t_count1 = int(before.split(';')[2].split(':')[1])
        t_count2 = int(cur.split(';')[2].split(':')[1])
        if t_count2 != t_count1:
            total_time += cal_gap(stamp1, stamp2)
    return total_time


def cal_point_accuracy(path, pc, error, type):
    low_bound = pc - error
    up_bound = pc + error
    satisfied_count = 0
    query_info = pd.read_csv(path, index_col=0)
    for index, row in query_info.iterrows():
        if type is 'cost':
            if low_bound <= row['total_cost'] <= up_bound:
                satisfied_count += 1
        else:
            if low_bound <= row['e_cardinality'] <= up_bound:
                satisfied_count += 1
    print("satisfied_count:{} total_count:{}".format(satisfied_count, query_info.shape[0]))
    return satisfied_count / (query_info.shape[0]-1)


def cal_range_accuracy(path, rc, type):
    low_bound = rc[0]
    up_bound = rc[1]
    satisfied_count = 0
    query_info = pd.read_csv(path, index_col=0)
    for index, row in query_info.iterrows():
        if type is 'cost':
            if low_bound <= row['total_cost'] <= up_bound:
                satisfied_count += 1
        else:
            if low_bound <= row['e_cardinality'] <= up_bound:
                satisfied_count += 1
    print("satisfied_count:{} total_count:{}".format(satisfied_count, query_info.shape[0]))
    return satisfied_count / (query_info.shape[0]-1)


def show_distribution(fpath, type):
    if not os.path.exists(fpath):
        print('path error')
    query_info = pd.read_csv(fpath, index_col=0)
    query_info['total_cost'] = query_info['total_cost'].apply(np.log1p)     # 防止log0
    query_info['e_cardinality'] = query_info['e_cardinality'].apply(np.log1p)
    if type == 'cost':
        plt.hist(query_info.total_cost,
                 bins=100,
                 color='steelblue',
                 edgecolor='k',
                 )
    else:
        plt.hist(query_info.e_cardinality,
                 bins=100,
                 color='steelblue',
                 edgecolor='k',
                 )
    plt.tick_params(top='off', right='off')
    plt.show()


def relative_error(e_value, t_value):
    return abs(e_value-t_value) / t_value


def trade_off_time(path):
    if not os.path.exists(path):
        print('no file')
        return
    time_gap = []
    with open(path, 'r') as f:
        log = f.readlines()
    # print(log)
    for index in range(1, len(log)):
        # print(log[index])
        cur = log[index]
        before = log[index-1]
        stamp1 = float(before.split(';')[0].split(':')[1])
        stamp2 = float(cur.split(';')[0].split(':')[1])
        time_gap.append(cal_gap(stamp1, stamp2, True))
    for i in range(1, len(time_gap)):
        time_gap[i] += time_gap[i-1]
    return time_gap


def temp_func(path):
    with open(path, 'r') as f:
        log = f.readlines()
    to_log = []
    last_record = -1
    for index in range(1, len(log)):
        cur = log[index]
        s_count = int(cur.split(';')[1].split(':')[1])
        if s_count % 10 == 0 and s_count != last_record:
            to_log.append(cur)
            last_record = s_count
    time_log = ''.join(to_log)
    with open(path, 'w') as f:
        f.write(time_log)


if __name__ == '__main__':
    # show_distribution('./cost/tpch/tpch100000_result', 'cost')
    # show_distribution('./cardinality/tpch/tpch10000_result', 'card')
    # path = '/home/lixizhang/learnSQL/sqlsmith/tpch/logfile/card_pc10000_N1000'
    # fpath = '/home/lixizhang/learnSQL/sqlsmith/tpch/statics/tpch100000'
    # tpath = fpath + '_result'
    # cal_file_e_info('tpch', fpath, tpath)

    # t = cal_time('/home/lixizhang/learnSQL/sqlsmith/imdbload/logfile/cost_rc1000_6000_N1000')
    # print(t)
    # path = './sqlsmith/imdbload/statics/imdbload10000_0_result'
    # pc = 100000000
    # error = pc * 0.1
    # print(cal_point_accuracy(path, pc, error, 'cost'))
    # rc = (1000, 2000)
    # print(cal_range_accuracy(path, rc, 'cost'))
    # print(cal_gap(1642611036.3752596, 1642300192.7910643))
    # temp_func('/Users/zhanglixi/PycharmProjects/learnSQL/ExtendEnv/tpch/cost_rc1000_6000_update_time')
    print(trade_off_time('/Users/zhanglixi/PycharmProjects/learnSQL/ExtendEnv/tpch/cost_rc1000_4000_nest_time'))

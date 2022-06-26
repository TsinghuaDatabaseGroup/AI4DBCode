import psycopg2
import math
import os
import random


def connect_server(dbname):
    db = psycopg2.connect(database=dbname, user="", password="", host="166.111.5.177", port="5433")
    cursor = db.cursor()
    return db, cursor


def random_sample_data(dbname_path, table_name, field, num, cursor):
    data = []
    sql = '''
            select min("{0}"), max("{0}"), count(*) from {1};
            '''.format('id', table_name)
    cursor.execute(sql)
    max_min_count = cursor.fetchall()[0]
    if max_min_count[2] < num:
        sql = '''
        select {0} from {1};
        '''.format(field, table_name)
        cursor.execute(sql)
        result = cursor.fetchall()
        for item in result:
            data.append(item[0])
    else:
        count = 0
        while count < num:
            index = random.randint(max_min_count[0], max_min_count[1])
            sql = '''
            select {0} from {1} where id={2}
            '''.format(field, table_name, index)
            cursor.execute(sql)
            result = cursor.fetchall()
            if len(result) != 0:
                data.append(result[0][0])
                count += 1
    table_path = dbname_path + '/' + table_name
    if not os.path.exists(table_path):
        os.mkdir(table_path)
    field_path = table_path + '/' + field + '.txt'
    with open(field_path, 'w') as f:
        for item in data:
            f.write(str(item).strip() + '\n')


def get_table_structure(cursor):
    """
    schema: {table_name: {field_name {'DataType', 'keytype'}}}
    :param cursor:
    :return:
    """
    cursor.execute('SELECT table_name FROM information_schema.tables WHERE table_schema = \'public\';')
    tables = cursor.fetchall()
    schema = {}
    for table_info in tables:
        table_name = table_info[0]
        sql = 'SELECT column_name FROM information_schema.columns WHERE table_name = \'' + table_name + '\';'
        cursor.execute(sql)
        columns = cursor.fetchall()
        schema[table_name] = []
        for col in columns:
            schema[table_name].append(col[0])
    return schema


def get_index_key(table_name, cursor, index_name):
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


def sample_max_min_data(dbname_path, table_name, field, cursor):
    sql = '''
    select min("{0}"), max("{0}") from {1};
    '''.format(field, table_name)
    cursor.execute(sql)
    sample_data = cursor.fetchall()[0]
    table_path = dbname_path + '/' + table_name
    if not os.path.exists(table_path):
        os.mkdir(table_path)
    field_path = table_path + '/' + field + '.txt'
    with open(field_path, 'w') as f:
        for data in sample_data:
            f.write(str(data).strip() + '\n')


def sample_index_data(dbname_path, table_name, field, num, cursor):
    sql = '''
        select min("{0}"), max("{0}") from {1};
        '''.format(field, table_name)
    cursor.execute(sql)
    max_min = cursor.fetchall()[0]
    if type(max_min[0]) == int:
        if max_min[1] < num:
            sample_data = range(max_min[0], max_min[1] + 1)
        else:
            step = math.ceil(max_min[1] / num)
            sample_data = range(max_min[0], max_min[1], step)
        table_path = dbname_path + '/' + table_name
        if not os.path.exists(table_path):
            os.mkdir(table_path)
        field_path = table_path + '/' + field + '.txt'
        with open(field_path, 'w') as f:
            for data in sample_data:
                f.write(str(data).strip() + '\n')
    else:
        random_sample_data(dbname_path, table_name, field, num, cursor)


def sample_database(dbname):
    db, cursor = connect_server(dbname)
    schema = get_table_structure(cursor)
    index_name = []
    dbname_path = os.path.abspath('..') + '/cardinality' + dbname
    if not os.path.exists(dbname_path):
        os.mkdir(dbname_path)
    # 收集索引信息
    for table_name in schema.keys():
        get_index_key(table_name, cursor, index_name)
    # sample data
    for table_name in schema.keys():
        for field in schema[table_name]:
            if field in index_name:
                sample_index_data(dbname_path, table_name, field, 10, cursor)
            else:
                sample_max_min_data(dbname_path, table_name, field, cursor)
    cursor.close()


# sample_database('imdbload')
sample_database('tpch')
import numpy as np
import pandas
import json
import os
import pymysql
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from configs import predictor_output_dim

query_types = ["insert", "delete", "update", "select"]


# base prediction model
def baseline_model(num_feature=len(query_types)):
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=num_feature, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(predictor_output_dim, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


class SqlParser:
    ###########################################################################
    # DML: select delete insert update      0 1 2 3
    # select {select_expr}
    # <modifier> (The first is default)
    # [ALL | DISTINCT | DISTINCTROW]
    # [0 | HIGH_PRIORITY], faster than update, with table-level lock
    # [0 | STRAIGHT_JOIN],
    # [0 | SQL_SMALL_RESULT | SQL_BIG_RESULT]
    # [0 | SQL_BUFFER_RESULT]
    # [SQL_CACHE | SQL_NO_CACHE]
    # [SQL_CALC_FOUND_ROWS]

    # OLTP (workload2vector)
    # select*w1 + sum(modifiers)*w2 + num({select_expr})*wl3        # 0.7 0.1 0.2
    # from [table]
    # [WHERE where_condition]   range join

    # OLTP (sql2vector)
    # cost-vector: [Aggregate, Nested Loop, Index Scan, Hash_Join]

    # Keywords
    # [GROUP BY {col_name | expr | position}]
    # [ASC | DESC], ...[WITH ROLLUP]
    # [HAVING where_condition]
    # [ORDER BY {col_name | expr | position}]
    # [ASC | DESC], ...

    # sum(group_table_scale(having)*wi) + order_cost*wi
    ###########################################################################

    def __init__(self, argus):

        self.resfile = os.path.join("scripts/") + "output.res"
        self.argus = argus
        self.conn = self.mysql_conn()
        self.tables = self.get_database_tables()
        self.query_encoding_map = {}
        ########### Convert from the sql statement to the sql vector
        #  directly read vector from a file (so a python2 script needs to run first!)
        #  sql_type * (num_events, C, aggregation, in-mem)
        #############################################################################################################################

        # query encoding features
        self.op_weight = {'oltp_point_select': 1, 'select_random_ranges': 2, 'oltp_delete': 3,
                          'oltp_insert': 4, 'bulk_insert': 5, 'oltp_update_index': 6,
                          'oltp_update_non_index': 7, }
        self.num_event = int(argus['num_event'])
        self.C = [10000]
        self.group_cost = 0
        self.in_mem = 0
        self.predict_sql_resource_value = None
        self.estimator = baseline_model(len(query_types) + len(self.tables))
        # Prepare Data
        fs = open("training-data/trainData_sql.txt", 'r')
        df = pandas.read_csv(fs, sep=' ', header=None)
        lt_sql = df.values
        # seperate into input X and output Y
        sql_op = lt_sql[:, 0]
        sql_X = lt_sql[:, 1:5]  # op_type   events  table_size
        sql_Y = lt_sql[:, 5:]

    def query_encoding(self, query):

        if not query:
            print("query is empty")
            return []

        if self.query_encoding_map.get(str(query), None):
            return self.query_encoding_map[str(query)]

        result = [0 for i in range(len(self.tables) + len(query_types))]
        # [0, 0, 0, 0, X, X, X..........]
        query_split_list = query.lower().split(" ")

        for index, query_type in enumerate(query_types):
            if query_type in query_split_list:
                result[index] = 1

        query_split_list = query.replace(",", "").split(" ")

        explain_format_fetchall = self.mysql_query("EXPLAIN FORMAT=JSON {};".format(query))
        if not explain_format_fetchall:
            print("explain_format_fetchall is empty, query: {}".format(query))
            return []
        explain_format = json.loads(explain_format_fetchall[0][0])
        explain_format_tables_list = self.get_explain_format_tables_list([], explain_format.get("query_block"), "table")
        for explain_format_table in explain_format_tables_list:
            explain_format_table_name = explain_format_table["table_name"]
            index = query_split_list.index(explain_format_table_name)
            if query_split_list[index - 1].lower() == "as":
                explain_format_table_name = query_split_list[index - 2]
            else:
                explain_format_table_name = query_split_list[index - 1]

            for index, table_name in enumerate(self.tables):
                if explain_format_table_name == table_name:
                    result[index + len(query_types)] = float(explain_format_table["cost_info"]["prefix_cost"])
                    continue
        self.query_encoding_map[str(query)] = result
        return result

    def predict_sql_resource(self, workload=[]):
        # Predict sql convert
        # inner_metric_change   np.array
        if self.predict_sql_resource_value is None:
            print("predict_sql_resource_value is None")
            exit()
        return self.predict_sql_resource_value
        # return self.estimator.predict(self.get_workload_encoding(
        #     workload))  # input : np.array([[...]])      (sq_type, num_events, C, aggregation, in-mem)
        # # output : np.array([[...]])

    def update(self):
        pass

    def mysql_conn(self):
        conn = pymysql.connect(
            host=self.argus["host"],
            user=self.argus["user"],
            passwd=self.argus["password"],
            port=int(self.argus["port"]),
            connect_timeout=30,
            charset='utf8')
        conn.select_db(self.argus["database"])
        return conn

    def close_mysql_conn(self):
        try:
            self.conn.close()
        except Exception as error:
            print("close mysqlconn: " + str(error))

    def mysql_query(self, sql):
        try:
            cursor = self.conn.cursor()
            count = cursor.execute(sql)
            if count == 0:
                result = 0
            else:
                result = cursor.fetchall()
            cursor.close()
            return result
        except Exception as error:
            print("mysql execute: " + str(error))
            return None

    def get_database_tables(self):
        # get all tables
        tables_fetchall = self.mysql_query(
            "select table_name from information_schema.tables where table_schema='{}';".format(self.argus["database"]))
        tables = []
        if not tables_fetchall:
            print("tables was not found")
            return
        for table in tables_fetchall:
            if table and table[0]:
                tables.append(table[0])
        print("get all tables success")
        return tables

    def get_explain_format_tables_list(self, values_list, json, key):
        if isinstance(json, dict):
            for item, values in json.items():
                if str(item) == str(key):
                    values_list.append(json.get(item))
                if isinstance(values, dict):
                    self.get_explain_format_tables_list(values_list, values, key=key)
                if isinstance(values, list):
                    self.get_explain_format_tables_list(values_list, values, key=key)
                else:
                    pass
        elif isinstance(json, list):
            for data in json:
                if isinstance(data, dict):
                    self.get_explain_format_tables_list(values_list, data, key)
        else:
            return values_list
        return values_list

    def get_workload_encoding(self, workload):
        queries_encoding = []
        for query in workload:
            queries_encoding.append(self.query_encoding(query))

        # [0, 0, 0, 0, X, X, X..........]
        workload_encoding = np.array([0 for i in range(len(self.tables) + len(query_types))])
        for query_encoding in queries_encoding:
            workload_encoding = workload_encoding + np.array(query_encoding)

        for i in range(len(query_types)):
            if workload_encoding[i] > 0:
                workload_encoding[i] = 1

        return workload_encoding.reshape(1, len(workload_encoding))

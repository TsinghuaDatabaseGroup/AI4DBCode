import numpy as np
import pandas

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler


# regression
def baseline_model(num_feature=4):
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=num_feature, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(65, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

class SqlParser:
    # DML: select delete insert update      0 1 2 3
    ###########################################################################
    # select
    # <modifier> (The first is default)
    # [ALL | DISTINCT | DISTINCTROW]
    # [0 | HIGH_PRIORITY], faster than update, with table-level lock
    # [0 | STRAIGHT_JOIN],
    # [0 | SQL_SMALL_RESULT | SQL_BIG_RESULT]
    # [0 | SQL_BUFFER_RESULT]
    # [SQL_CACHE | SQL_NO_CACHE]
    # [SQL_CALC_FOUND_ROWS]

    # {select_expr}

    # select*w1 + sum(modifiers)*w2 + num({select_expr})*wl3
    # 0.7 0.1 0.2

    # from [table]
    # [PARTITION partition_list]
    # [WHERE where_condition]   range join

    # Student(sid, sname, score, grade) Class(cid, grade) => [0, 1, 0, 1],[0, 1]        <take min for join; >

    # sum(table_scale(where: point=1 range=[])*wi) + 0*(partition)*wi

    # [GROUP BY {col_name | expr | position}]
    # [ASC | DESC], ...[WITH ROLLUP]
    # [HAVING where_condition]
    # [ORDER BY {col_name | expr | position}]
    # [ASC | DESC], ...

    # sum(group_table_scale(having)*wi) + order_cost*wi
    ###########################################################################

    def __init__(self, cur_op='oltp_read_write.lua', num_event=1000, p_r_range=0.6, p_u_index=0.2, p_i=0.1, p_d=0.1):

        # sysbench: single table operations
        self.cmd = "sysbench --test=oltp --oltp-table-size=5000 " + " --num-threads=5 --max-requests=" + str(
            num_event) + " --mysql-host=127.0.0.1 --mysql-user='root' --mysql-password='db2019' --mysql-port=3306 --db-ps-mode=disable --mysql-db='test' \
                  --oltp-simple-ranges=" + str(int(num_event * p_r_range)) + " --oltp-index-updates=" + str(int(num_event * p_u_index))

        # join: do join among imdb tables
        self.join_cmd = ''

        # sql_type_weight * (D, [opi])
        #### sql_statement
        #### model -> self.sql_vector
        ####

        ########### Convert from the sql statement to the sql vector
        #  directly read vector from a file (so a python2 script needs to run first!)
        #  sql_type * (num_events, C, aggregation, in-mem)
        #############################################################################################################################
        self.op_weight = {'oltp_point_select.lua': 1, 'select_random_ranges.lua': 2, 'oltp_delete.lua': 3,
                          'oltp_insert.lua': 4, 'bulk_insert.lua': 5, 'oltp_update_index.lua': 6,
                          'oltp_update_non_index.lua': 7, }

        self.cur_op = cur_op
        self.num_event = num_event
        self.C = [10000]
        self.group_cost = 0
        self.in_mem = 0

        # sql representation
        if self.cur_op == "oltp_read_write.lua":
            op_weight = self.op_weight['select_random_ranges.lua'] * p_r_range + self.op_weight[
                'oltp_update_index.lua'] * p_u_index + self.op_weight['oltp_insert.lua'] * p_i + self.op_weight[
                            'oltp_delete.lua'] * p_d
            print("op_weight:%f" % op_weight)
        else:
            op_weight = self.op_weight[self.cur_op]

        self.sql_vector = np.array([[self.num_event, self.C[0], self.group_cost, self.in_mem]])
        self.sql_vector = np.array([[o * op_weight for o in self.sql_vector[0]]])
        ################################################################################################################################




        # Create the sql convert model
        # Fetch the data
        fs = open("trainData_sql.txt", 'r')
        df = pandas.read_csv(fs, sep=' ', header=None)
        lt_sql = df.values
        # seperate into input X and output Y
        sql_op = lt_sql[:, 0]
        sql_X = lt_sql[:, 1:5]  # op_type   events  table_size
        sql_Y = lt_sql[:, 5:]
        print(sql_Y[0])

        for i, s in enumerate(sql_op):
            s = s + 1
            sql_X[i][0] = sql_X[i][0] * s
            sql_X[i][1] = sql_X[i][1] * s
            sql_X[i][2] = sql_X[i][2] * s
            sql_X[i][3] = sql_X[i][3] * s


        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(sql_X)
        # X_test = X_train[50:]
        # X_train = X_train[:50]

        sc_Y = StandardScaler()
        Y_train = sc_Y.fit_transform(sql_Y)
        Y_test = Y_train[50:]
        # Y_train = Y_train[:50]

        # Create the sql convert model
        # evaluate model with standardized dataset
        seed = 7
        np.random.seed(seed)
        # estimators = []
        # estimators.append(('standardize', StandardScaler()))
        # estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=50, verbose=0)))
        self.estimator = KerasRegressor(build_fn=baseline_model, epochs=1000, batch_size=50, verbose=1)  # epochs
        self.estimator.fit(X_train, Y_train)

    def predict_sql_resource(self):
        # Predict sql convert
        # inner_metric_change   np.array
        return self.estimator.predict(
            self.sql_vector)  # input : np.array([[...]])      (sq_type, num_events, C, aggregation, in-mem)
        # output : np.array([[...]])

    def update(self):
        pass

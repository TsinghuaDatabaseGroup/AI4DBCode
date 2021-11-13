import datetime
import subprocess
from collections import deque
import numpy as np

import pymysql
import pymysql.cursors as pycursor

import gym
from gym import spaces
from gym.utils import seeding

from sql2resource import SqlParser

from configs import knob_config
import time
from run_job import run_job


# fetch all the knobs from the prepared configuration info


class Database:
    def __init__(self, argus):
        self.argus = argus
        # self.internal_metric_num = 13 # 13(state) + cumulative()
        self.external_metric_num = 2  # [throughput, latency]           # num_event / t
        self.max_connections_num = None
        self.knob_names = [knob for knob in knob_config]
        print("knob_names:", self.knob_names)
        self.knob_num = len(knob_config)
        self.internal_metric_num = 65  # default system metrics enabled in metric_innodb
        self.max_connections()
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            sql = "SELECT count FROM INFORMATION_SCHEMA.INNODB_METRICS where status='enabled'"
            cursor.execute(sql)
            result = cursor.fetchall()
            self.internal_metric_num = len(result)
            cursor.close()
            conn.close()
        except Exception as err:
            print("execute sql error:", err)

    def _get_conn(self):
        conn = pymysql.connect(host=self.argus['host'],
                               port=int(self.argus['port']),
                               user=self.argus['user'],
                               password=self.argus['password'],
                               db='INFORMATION_SCHEMA',
                               connect_timeout=36000,
                               cursorclass=pycursor.DictCursor)
        return conn

    def fetch_internal_metrics(self):
        ######### observation_space
        #         State_status
        # [lock_row_lock_time_max, lock_row_lock_time_avg, buffer_pool_size,
        # buffer_pool_pages_total, buffer_pool_pages_misc, buffer_pool_pages_data, buffer_pool_bytes_data,
        # buffer_pool_pages_dirty, buffer_pool_bytes_dirty, buffer_pool_pages_free, trx_rseg_history_len,
        # file_num_open_files, innodb_page_size]
        #         Cumulative_status
        # [lock_row_lock_current_waits, ]
        '''
        sql = "select count from INNODB_METRICS where name='lock_row_lock_time_max' or name='lock_row_lock_time_avg'\
        or name='buffer_pool_size' or name='buffer_pool_pages_total' or name='buffer_pool_pages_misc' or name='buffer_pool_pages_data'\
        or name='buffer_pool_bytes_data' or name='buffer_pool_pages_dirty' or name='buffer_pool_bytes_dirty' or name='buffer_pool_pages_free'\
        or name='trx_rseg_history_len' or name='file_num_open_files' or name='innodb_page_size'"
        '''
        state_list = np.array([])
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            sql = "SELECT count FROM INFORMATION_SCHEMA.INNODB_METRICS where status='enabled'"
            cursor.execute(sql)
            result = cursor.fetchall()
            for s in result:
                state_list = np.append(state_list, [s['count']])
            cursor.close()
            conn.close()
        except Exception as error:
            print(error)

        return state_list

    def fetch_knob(self):
        state_list = np.append([], [])
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            sql = "select"
            for i, knob in enumerate(self.knob_names):
                sql = sql + ' @@' + knob
                if i < self.knob_num - 1:
                    sql = sql + ', '
            # print("fetch_knob:", sql)
            cursor.execute(sql)
            result = cursor.fetchall()
            for i in range(self.knob_num):
                state_list = np.append(state_list, result[0]["@@%s" % self.knob_names[i]])
            cursor.close()
            conn.close()
        except Exception as error:
            print("fetch_knob Error:", error)
        return state_list

    def max_connections(self):
        # if not self.max_connections_num:
        if 1:
            try:
                conn = self._get_conn()
                cursor = conn.cursor()
                sql = "show global variables like 'max_connections';"
                cursor.execute(sql)
                self.max_connections_num = int(cursor.fetchone()["Value"])
                cursor.close()
                conn.close()
            except Exception as error:
                print(error)
        return self.max_connections_num

    def change_knob_nonrestart(self, actions):
        try:
            conn = self._get_conn()
            for i in range(self.knob_num):
                cursor = conn.cursor()
                if self.knob_names[i] == 'max_connections':
                    self.max_connections_num = actions[i]
                sql = 'set global %s=%d' % (self.knob_names[i], actions[i])
                cursor.execute(sql)
                # print(f"修改参数-{self.knob_names[i]}:{actions[i]}")
                conn.commit()
            conn.close()
            return 1
        except Exception as error:
            conn.close()
            print("change_knob_nonrestart error：", error)
            return 0


# Define the environment
class Environment(gym.Env):

    def __init__(self, db, argus):

        self.db = db

        self.parser = SqlParser(argus)

        self.state_num = db.internal_metric_num
        self.action_num = db.knob_num
        self.timestamp = int(time.time())

        # pfs = open('training-results/res_predict-' + str(self.timestamp), 'a')
        # pfs.write("%s\t%s\t%s\n" % ('iteration', 'throughput', 'latency'))
        # pfs.close()
        #
        # rfs = open('training-results/res_random-' + str(self.timestamp), 'a')
        # rfs.write("%s\t%s\t%s\n" % ('iteration', 'throughput', 'latency'))
        # rfs.close()

        ''' observation dim = system metric dim + query vector dim '''
        self.score = 0  # accumulate rewards

        self.o_dim = db.internal_metric_num + len(self.db.fetch_internal_metrics())
        self.o_low = np.array([-1e+10] * self.o_dim)
        self.o_high = np.array([1e+10] * self.o_dim)

        self.observation_space = spaces.Box(low=self.o_low, high=self.o_high, dtype=np.float32)
        # part 1: current system metric
        self.state = db.fetch_internal_metrics()
        # print("Concatenated state:")
        # part 2: predicted system metric after executing the workload
        self.workload = argus["workload"]

        # TODO: 打开训练predict的方法后，此方法注释
        ################################################################################
        state0 = self.db.fetch_internal_metrics()
        self.preheat()
        state1 = self.db.fetch_internal_metrics()
        try:
            if self.parser.predict_sql_resource_value is None:
                self.parser.predict_sql_resource_value = state1 - state0
        except Exception as error:
            print("get predict_sql_resource_value error:", error)
        ################################################################################

        self.state = np.append(self.parser.predict_sql_resource(self.workload), self.state)

        ''' action space '''
        # Offline
        # table_open_cache(1), max_connections(2), innodb_buffer_pool_instances(4),
        # innodb_log_files_in_group(5), innodb_log_file_size(6), innodb_purge_threads(7), innodb_read_io_threads(8)
        # innodb_write_io_threads(9),
        # Online
        # innodb_buffer_pool_size(3), max_binlog_cache_size(10), binlog_cache_size(11)
        # 1 2 3 11
        # exclude
        # innodb_file_per_table, skip_name_resolve, binlog_checksum,
        # binlog_format(dynamic, [ROW, STATEMENT, MIXED]),

        calculate_knobs = []
        infer_knobs = []
        for k in knob_config.items():
            if k[1]['type'] == 'infer':
                infer_knobs.append(k)
            else:
                calculate_knobs.append(k)
        self.knob_num = len(knob_config)
        # self.a_low = np.array([knob[1]['min_value']/knob[1]['length'] for knob in list(knob_config.items())[:db.knob_num]])
        self.a_low = np.array([knob[1]['min_value'] / knob[1]['length'] for knob in infer_knobs])
        # self.a_high = np.array([knob[1]['max_value']/knob[1]['length'] for knob in list(knob_config.items())[:db.knob_num]])
        self.a_high = np.array([knob[1]['max_value'] / knob[1]['length'] for knob in infer_knobs])
        # self.length = np.array([knob[1]['length'] for knob in list(knob_config.items())[:db.knob_num]])
        self.length = np.array([knob[1]['length'] * 1.0 for knob in infer_knobs])
        self.action_space = spaces.Box(low=self.a_low, high=self.a_high, dtype=np.float32)
        self.default_action = self.a_low
        self.mem = deque(maxlen=int(argus['maxlen_mem']))  # [throughput, latency]
        self.predicted_mem = deque(maxlen=int(argus['maxlen_predict_mem']))
        self.knob2pos = {knob: i for i, knob in enumerate(knob_config)}
        self.seed()
        self.start_time = datetime.datetime.now()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def execute_command(self):
        self.db.max_connections()
        # print(self.db.max_connections_num) 
        if self.parser.argus['thread_num_auto'] == '0':
            thread_num = int(self.parser.argus['thread_num'])
        else:
            thread_num = int(self.db.max_connections_num) - 1
        run_job(thread_num, self.workload, self.parser.resfile)

    def preheat(self):
        self.execute_command()

    def fetch_action(self):
        while True:
            state_list = self.db.fetch_knob()
            if list(state_list):
                break
            time.sleep(5)
        return state_list

    # new_state, reward, done,
    def step(self, u, isPredicted, iteration, action_tmp=None):
        flag = self.db.change_knob_nonrestart(u)

        # if failing to tune knobs, give a high panlty
        if not flag:
            return self.state, -10e+4, self.score, 1

        self.execute_command()
        throughput, latency = self._get_throughput_latency()
        # ifs = open('fl1', 'r')
        # print(str(len(self.mem)+1)+"\t"+str(throughput)+"\t"+str(latency))
        cur_time = datetime.datetime.now()
        interval = (cur_time - self.start_time).seconds
        self.mem.append([throughput, latency])
        # 2 refetch state
        self._get_obs()
        # 3 cul reward(T, L)
        reward = self._calculate_reward(throughput, latency)
        '''
        reward = 0
        for i in range(u.shape[0]):
            tmp = u[i] / self.a_high[i]
            reward+=tmp
        # print("Performance: %d\t%f\t%f\t%f\t%ds" % (len(self.mem) + 1, throughput, latency, reward, interval))
        if isPredicted:
            self.predicted_mem.append([len(self.predicted_mem), throughput, latency, reward])
            if len(self.predicted_mem)%10 == 0:
                print("Predict List")
                print(self.predicted_mem)
       '''

        action = self.fetch_action()

        if isPredicted:
            self.predicted_mem.append([len(self.predicted_mem), throughput, latency, reward])

            # print("Predict %d\t%f\t%f\t%f\t%ds" % (len(self.mem) + 1, throughput, latency, reward, interval))

            pfs = open('training-results/res_predict-' + str(self.timestamp), 'a')
            pfs.write("%d\t%f\t%f\n" % (iteration, throughput, latency))
            pfs.close()

            fetch_knob = open('training-results/fetch_knob_predict-' + str(self.timestamp), 'a')
            fetch_knob.write(f"{str(iteration)}\t{str(list(action))}\n")
            fetch_knob.close()

            action_write = open('training-results/action_test_predict-' + str(self.timestamp), 'a')
            action_write.write(f"{str(iteration)}\t{str(list(u))}\n")
            action_write.write(f"{str(iteration)}\t{str(list(action_tmp))}\n")
            action_write.close()

            self.score = self.score + reward

        else:
            # print("Random %d\t%f\t%f\t%f\t%ds" % (len(self.mem) + 1, throughput, latency, reward, interval))

            rfs = open('training-results/res_random-' + str(self.timestamp), 'a')
            rfs.write("%d\t%f\t%f\n" % (iteration, throughput, latency))
            rfs.close()

            action_write = open('training-results/action_random-' + str(self.timestamp), 'a')
            action_write.write(f"{str(iteration)}\t{str(list(u))}\n")
            action_write.close()

            fetch_knob = open('training-results/fetch_knob_random-' + str(self.timestamp), 'a')
            fetch_knob.write(f"{str(iteration)}\t{str(list(action))}\n")
            fetch_knob.close()

        return self.state, reward, self.score, throughput

    def _get_throughput_latency(self):
        with open(self.parser.resfile, 'r') as f:
            try:
                for line in f.readlines():
                    a = line.split()
                    if len(a) > 1 and 'avg_qps(queries/s):' == a[0]:
                        throughput = float(a[1])
                    if len(a) > 1 and 'avg_lat(s):' == a[0]:
                        latency = float(a[1])
            finally:
                f.close()
            # print("throughput:{} \n latency:{}".format(throughput, latency))
            return throughput, latency

    def _calculate_reward(self, throughput, latency):
        if len(self.mem) != 0:
            dt0 = (throughput - self.mem[0][0]) / self.mem[0][0]
            dt1 = (throughput - self.mem[len(self.mem) - 1][0]) / self.mem[len(self.mem) - 1][0]
            if dt0 >= 0:
                rt = ((1 + dt0) ** 2 - 1) * abs(1 + dt1)
            else:
                rt = -((1 - dt0) ** 2 - 1) * abs(1 - dt1)

            dl0 = -(latency - self.mem[0][1]) / self.mem[0][1]

            dl1 = -(latency - self.mem[len(self.mem) - 1][1]) / self.mem[len(self.mem) - 1][1]

            if dl0 >= 0:
                rl = ((1 + dl0) ** 2 - 1) * abs(1 + dl1)
            else:
                rl = -((1 - dl0) ** 2 - 1) * abs(1 - dl1)

        else:  # initial action
            rt = 0
            rl = 0
        reward = 1 * rl + 9 * rt
        return reward

    def _get_obs(self):
        self.state = self.db.fetch_internal_metrics()
        self.state = np.append(self.parser.predict_sql_resource(self.workload), self.state)
        return self.state

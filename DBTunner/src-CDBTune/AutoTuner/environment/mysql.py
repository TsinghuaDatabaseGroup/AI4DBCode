# -*- coding: utf-8 -*-
"""
description: MySQL Environment
"""

import re
import os
import time
import math
import datetime
import json
import threading
import MySQLdb
import numpy as np
import configs
import utils
import knobs
import requests
import psutil
from base import *
from db import database
from utils import *

logger = cdb_logger



class Status(object):
    OK = 'OK'
    FAIL = 'FAIL'
    RETRY = 'RETRY'


class MySQLEnv(object):

    def __init__(self, wk_type='read', method='sysbench',  alpha=1.0, beta1=0.5, beta2=0.5, time_decay1=1.0, time_decay2=1.0):

        self.db_info = None
        self.wk_type = wk_type
        self.score = 0.0
        self.steps = 0
        self.terminate = False
        self.last_external_metrics = None
        self.default_externam_metrics = None

        self.method = method
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.time_decay_1 = time_decay1
        self.time_decay_2 = time_decay2
       

    @staticmethod
    def _get_external_metrics(path, method='sysbench'):

        def parse_tpcc(file_path):
            with open(file_path) as f:
                lines = f.read()
            temporal_pattern = re.compile(".*?trx: (\d+.\d+), 95%: (\d+.\d+), 99%: (\d+.\d+), max_rt:.*?")
            temporal = temporal_pattern.findall(lines)
            tps = 0
            latency = 0
            qps = 0

            for i in temporal[-10:]:
                tps += float(i[0])
                latency += float(i[2])
            num_samples = len(temporal[-10:])
            tps /= num_samples
            latency /= num_samples
            # interval
            tps /= 1
            return [tps, latency, tps]

        def parse_sysbench(file_path):
            with open(file_path) as f:
                lines = f.read()
            temporal_pattern = re.compile(
                "tps: (\d+.\d+) qps: (\d+.\d+) \(r/w/o: (\d+.\d+)/(\d+.\d+)/(\d+.\d+)\)" 
                " lat \(ms,95%\): (\d+.\d+) err/s: (\d+.\d+) reconn/s: (\d+.\d+)")
            temporal = temporal_pattern.findall(lines)
            tps = 0
            latency = 0
            qps = 0

            for i in temporal[-10:]:
                tps += float(i[0])
                latency += float(i[5])
                qps += float(i[1])
            num_samples = len(temporal[-10:])
            tps /= num_samples
            qps /= num_samples
            latency /= num_samples
            return [tps, latency, qps]

        if method == 'sysbench':
            result = parse_sysbench(path)
        elif method == 'tpcc':
            result = parse_tpcc(path)
        else:
            result = parse_sysbench(path)
        return result

    def _get_internal_metrics(self, internal_metrics):
        """
        Args:
            internal_metrics: list,
        Return:

        """
        _counter = 0
        _period = 5
        count = 160/5

        def collect_metric(counter):
            counter += 1
            timer = threading.Timer(_period, collect_metric, (counter,))
            

            timer.start()
            db = database(self.db_info["host"],
                    self.db_info["port"],self.db_info["user"],
                    self.db_info["password"],
                    "sbtest",
                    )
            if counter >= count:
                timer.cancel()
            try:
                data = utils.get_metrics(db)
                internal_metrics.append(data)
            except Exception as err:
                logger.info("[GET Metrics]Exception:" ,err) 

        collect_metric(_counter)

        return internal_metrics

    def _post_handle(self, metrics):
        result = np.zeros(self.num_metric)

        def do(metric_name, metric_values):
            metric_type = utils.get_metric_type(metric_name)
            if metric_type == 'counter':
                return float(metric_values[-1] - metric_values[0])
            else:
                return float(sum(metric_values))/len(metric_values)

        keys = metrics[0].keys()

        keys.sort()
        for idx in xrange(len(keys)):
            key = keys[idx]
            data = [x[key] for x in metrics]
            result[idx] = do(key, data)
        return result

    def initialize(self):
        """Initialize the mysql instance environment
        """
        pass

    def eval(self, knob):
        """ Evaluate the knobs
        Args:
            knob: dict, mysql parameters
        Returns:
            result: {tps, latency}
        """
        flag = self._apply_knobs(knob)
        if not flag:
            return {"tps": 0, "latency": 0}

        external_metrics, _ = self._get_state(knob, method=self.method)
        return {"tps": external_metrics[0],
                "latency": external_metrics[1]}

    def _get_best_now(self, filename):
        with open(self.best_result) as f:
            lines = f.readlines()
        best_now = lines[0].split(',')
        return [float(best_now[0]), float(best_now[1]), float(best_now[0])]

    def record_best(self, external_metrics):
        best_flag = False
        if os.path.exists(self.best_result):
            tps_best = external_metrics[0]
            lat_best = external_metrics[1]
            rate = 0
            if int(lat_best) != 0:
                rate = float(tps_best)/lat_best
                with open(self.best_result) as f:
                    lines = f.readlines()
                best_now = lines[0].split(',')
                rate_best_now = float(best_now[0])/float(best_now[1])
                if rate > rate_best_now:
                    best_flag = True
                    with open(self.best_result, 'w') as f:
                        f.write(str(tps_best) + ',' + str(lat_best) + ',' + str(rate))
        else:
            file = open(self.best_result, 'wr')
            tps_best = external_metrics[0]
            lat_best = external_metrics[1]
            rate = 0
            if int(lat_best) == 0 :
                rate = 0
            else:
                rate = float(tps_best)/lat_best
            file.write(str(tps_best) + ',' + str(lat_best) + ',' + str(rate))
        return best_flag

    def step(self, knob):
        """step
        """
        restart_time = utils.time_start()
        flag = self._apply_knobs(knob)
        restart_time = utils.time_end(restart_time)
        if not flag:
            return -10000000.0, np.array([0] * self.num_metric), True, self.score - 10000000, [0, 0, 0], restart_time
        s = self._get_state(knob, method=self.method)
        if s is None:
            return -10000000.0, np.array([0] * self.num_metric), True, self.score - 10000000, [0, 0, 0], restart_time
        external_metrics, internal_metrics = s

        reward = self._get_reward(external_metrics)
        flag = self.record_best(external_metrics)
        if flag == True:
            logger.info('Better performance changed!')
        else:
            logger.info('Performance remained!')
        #get the best performance so far to calculate the reward
        best_now_performance = self._get_best_now(self.best_result)
        self.last_external_metrics = best_now_performance

        next_state = internal_metrics
        terminate = self._terminate()
        knobs.save_knobs(
            knob = knob,
            metrics = external_metrics,
            instance=self.db_info,
            task_id=self.task_id
        )
        return reward, next_state, terminate, self.score, external_metrics, restart_time

    def setting(self, knob):
        self._apply_knobs(knob)
    
    def _get_state(self, knob, method='sysbench'):
        """Collect the Internal State and External State
        """
        timestamp = int(time.time())
        
        filename = CONST.FILE_LOG_SYSBENCH % (self.task_id,timestamp)
        
        internal_metrics = []
        self._get_internal_metrics(internal_metrics)
        #calculate the sysbench time automaticly
        if knob['innodb_buffer_pool_size'] < 161061273600:
            time_sysbench = 150
        else:
            time_sysbench = int(knob['innodb_buffer_pool_size']/1024.0/1024.0/1024.0/1.1)
        if self.method == 'sysbench':
            a = time.time()
            _sys_run = "bash %s %s %s %d %s %s %d %d %d %d %s" % (CONST.BASH_SYSBENCH,
                self.wk_type,self.db_info['host'],self.db_info['port'],self.db_info['user'],self.db_info['password'],
                self.db_info['tables'],self.db_info['table_size'],self.threads, time_sysbench, filename)

            logger.info("sysbench started")
            logger.info(_sys_run)
            osrst = os.system(_sys_run)
            logger.info("sysbench ended")

            a = time.time() - a
    
            if osrst != 0 or a < 50:
                os_quit(Err.RUN_SYSYBENCH_FAILED)

        elif self.method == 'tpcc':
            def kill_tpcc():
                def _filter_pid(x):
                    try:
                        x = psutil.Process(x)
                        if x.name() == 'tpcc_start':
                            return True
                        return False
                    except:
                        return False
                pids = psutil.pids()
                tpcc_pid = filter(_filter_pid, pids)
                logger.info(tpcc_pid) 
                for tpcc_pid_i in tpcc_pid:
                    os.system('kill %s' % tpcc_pid_i)

            timer = threading.Timer(170, kill_tpcc)
            timer.start()
            os.system('bash %s %s %d %s %s' % (CONST.BASH_TPCC,
                self.db_info['host'],self.db_info['port'],self.db_info['passwd'],filename))
            time.sleep(10)

        external_metrics = self._get_external_metrics(filename, method)
        internal_metrics = self._post_handle(internal_metrics)

        return external_metrics, internal_metrics

    def _apply_knobs(self, knob):
        """ Apply Knobs to the instance
        """
        pass

    @staticmethod
    def _calculate_reward(delta0, deltat):

        if delta0 > 0:
            _reward = ((1+delta0)**2-1) * math.fabs(1+deltat)
        else:
            _reward = - ((1-delta0)**2-1) * math.fabs(1-deltat)

        if _reward > 0 and deltat < 0:
            _reward = 0
        return _reward

    def _get_reward(self, external_metrics):
        """
        Args:
            external_metrics: list, external metric info, including `tps` and `qps`
        Return:
            reward: float, a scalar reward
        """
        logger.info('*****************************')
        logger.info(external_metrics)
        logger.info(self.default_externam_metrics)
        logger.info(self.last_external_metrics)
        logger.info('*****************************')
        # tps
        delta_0_tps = float((external_metrics[0] - self.default_externam_metrics[0]))/self.default_externam_metrics[0]
        delta_t_tps = float((external_metrics[0] - self.last_external_metrics[0]))/self.last_external_metrics[0]

        tps_reward = self._calculate_reward(delta_0_tps, delta_t_tps)

        # latency
        delta_0_lat = float((-external_metrics[1] + self.default_externam_metrics[1])) / self.default_externam_metrics[1]
        delta_t_lat = float((-external_metrics[1] + self.last_external_metrics[1])) / self.last_external_metrics[1]

        lat_reward = self._calculate_reward(delta_0_lat, delta_t_lat)
        
        reward = tps_reward * 0.4 + 0.6 * lat_reward
        self.score += reward
        logger.info('$$$$$$$$$$$$$$$$$$$$$$')
        logger.info(tps_reward)
        logger.info(lat_reward)
        logger.info(reward)
        logger.info('$$$$$$$$$$$$$$$$$$$$$$')
        if reward > 0:
            reward = reward*1000000
        return reward

    def _terminate(self):
        return self.terminate


class TencentServer(MySQLEnv):
    """ Build an environment in Tencent Cloud
    """

    def __init__(self,  instance, task_detail,model_detail,host):
        """Initialize `TencentServer` Class
        Args:
            instance_name: str, mysql instance name, get the database infomation
        """
        MySQLEnv.__init__(self, task_detail["rw_mode"])
        # super(MySQLEnv, self).__init__()
        self.wk_type = task_detail["rw_mode"]
        self.score = 0.0
        self.num_metric = model_detail["dimension"]
        self.steps = 0
        self.task_id = task_detail["task_id"]
        self.terminate = False
        self.last_external_metrics = None
        self.db_info = instance
        self.host = host
        self.alpha = 1.0
        self.method = task_detail["run_mode"]
        self.best_result = CONST.FILE_LOG_BEST % self.task_id
        self.threads = task_detail["threads"]

        knobs.init_knobs(instance,model_detail["knobs"])
        self.default_knobs = knobs.get_init_knobs()

    def _set_params(self, knob):
        """ Set mysql parameters by send GET requests to server
        Args:
            knob: dict, mysql parameters
        Return:
            workid: str, point to the setting process
        Raises:
            Exception: setup failed
        """
        
        instance_id = self.db_info['instance_id']

        data = dict()
        data["instanceid"] = instance_id
        data["operator"] = "cdbtune"
        para_list = []
        for kv in knob.items():
            para_list.append({"name": str(kv[0]), "value": str(kv[1])})
        data["para_list"] = para_list
        data = json.dumps(data)
        data = "data=" + data
        
        response = parse_json(CONST.URL_SET_PARAM % self.host, data)
        
        err = response['errno']
        if err != 0:
            raise Exception("SET UP FAILED: {}".format(err))

        # if restarting isn't needed, workid should be ''
        workid = response.get('workid', '')

        return workid

    def _get_setup_state(self, workid):
        """ Set mysql parameters by send GET requests to server
        Args:
            workid: str, point to the setting process
        Return:
            status: str, setup status (running, undoed)
        Raises:
            Exception: get state failed
        """
        instance_id = self.db_info['instance_id']

        data = dict()
        data['instanceid'] = instance_id
        data['operator'] = "cdbtune"
        data['workid'] = workid
        data = json.dumps(data)
        data = 'data=' + data

        response = parse_json(CONST.URL_QUERY_SET_PARAM % self.host, data)

        err = response['errno']
        status = response['status']

        if err != 0:
            # raise Exception("GET STATE FAILED: {}".format(err))
            return "except"

        return status

    def initialize(self):
        """ Initialize the environment when an episode starts
        Returns:
            state: np.array, current state
        """
        self.score = 0.0
        self.last_external_metrics = []
        self.steps = 0
        self.terminate = False

        flag = self._apply_knobs(self.default_knobs)
        i = 0
        while not flag:
            if i >= 2:
                logger.info("Initialize: {} times ....".format(i))
                os_quit(Err.SET_MYSQL_PARAM_FAILED)
            flag = self._apply_knobs(self.default_knobs)
            i += 1
        

        external_metrics, internal_metrics = self._get_state(knob = self.default_knobs, method=self.method)
        if os.path.exists(self.best_result):
            if os.path.getsize(self.best_result):
                with open(self.best_result) as f:
                    lines = f.readlines()
                best_now = lines[0].split(',')
                self.last_external_metrics = [float(best_now[0]), float(best_now[1]), float(best_now[0])]
        else:
            self.last_external_metrics = external_metrics
        self.default_externam_metrics = external_metrics

        state = internal_metrics
        knobs.save_knobs(
            self.default_knobs,
            metrics=external_metrics,
            instance=self.db_info,
            task_id=self.task_id
        )
        return state, external_metrics

    def _apply_knobs(self, knob):
        """ Apply the knobs to the mysql
        Args:
            knob: dict, mysql parameters
        Returns:
            flag: status, ['OK', 'FAIL', 'RETRY']
        """
        self.steps += 1
        i = 2
        workid = ''
        while i >= 0:
            try:
                workid = self._set_params(knob=knob)
            except Exception as e:
                logger.error("{}".format(e.message))
            else:
                break
            time.sleep(20)
            i -= 1
        if i == -1:
            logger.error("Failed too many times!")
            os_quit(Err.SET_MYSQL_PARAM_FAILED)
            return False

        # set parameters without restarting, sleep 20 seconds
        if len(workid) == 0:
            time.sleep(20)
            return True

        logger.info("Finished setting parameters..")
        steps = 0
        max_steps = 500

        status = self._get_setup_state(workid=workid)
        while status in ['not_start','running', 'pause', 'paused', 'except'] and steps < max_steps:
            time.sleep(15)
            status = self._get_setup_state(workid=workid)
            steps += 1

        logger.info("Out of Loop, status: {} loop step: {}".format(status, steps))

        if status == 'normal_finish':
            return True

        if status in ['notstart', 'undoed', 'undo'] or steps > max_steps:
            time.sleep(15)
            params = ''
            for key in knob.keys():
                params += ' --%s=%s' % (key, knob[key])
            logger.error("set param failed: {}".format(params))
            return False

        return False

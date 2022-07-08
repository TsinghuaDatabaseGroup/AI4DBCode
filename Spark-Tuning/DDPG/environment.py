""" 
@author: jdfeng
@contact: jdefeng@stu.xmu.edu.cn
@software: PyCharm 
@file: main.py 
@create: 2020/10/9 17:52 
"""
import gym
import numpy as np
from gym import spaces
from run_action import run_bench
import pydoop.hdfs as hdfs
import json
import time

# action
a_low = np.array([1, 1, 1, 1, 1, 1, 1, 0, 4, 1, 1, 1, 0, 1, 0])
a_high = np.array([8, 8, 9, 8, 8, 8, 8, 4, 8, 4, 9, 4, 1, 8, 1])

# state
o_low = np.array([0, 0, 0, 0, 0, 0])
o_high = np.array([10, 10, 150000000, 1000000, 1000000, 1000000])

last_log = ""

spark_conf_names = ['spark.default.parallelism', 'spark.driver.cores', 'spark.driver.memory',
                    'spark.driver.maxResultSize',
                    'spark.executor.instances', 'spark.executor.cores', 'spark.executor.memory',
                    'spark.executor.memoryOverhead',
                    'spark.files.maxPartitionBytes', 'spark.memory.fraction', 'spark.memory.storageFraction',
                    'spark.reducer.maxSizeInFlight',
                    'spark.shuffle.compress', 'spark.shuffle.file.buffer', 'spark.shuffle.spill.compress']


def get_rs(state, action):
    # find log on hdfs
    his_file_list = hdfs.ls("/history/")
    log_path = his_file_list[-1]
    global last_log

    if last_log == log_path:
        return -10000, state

    last_log = log_path
    print(last_log)
    log_file = hdfs.open(log_path, 'rt', encoding='utf-8')

    # get reward and state
    task_durations = []
    gc_times = []
    input_sizes = []
    records = []
    shuffle_reads = []
    shuffle_writes = []

    start_timestamp = None
    end_timestamp = None
    for line in log_file:
        try:
            line_json = json.loads(line)
        except:
            print('json错误')
            continue

        # 输出15个参数，对比一下action，看处理的日志文件有没有错
        if line_json['Event'] == 'SparkListenerEnvironmentUpdate':
            spark_props = line_json['Spark Properties']
            s = ''
            for conf_name in spark_conf_names:
                s = s + spark_props[conf_name] + ", "

            print()
            print(s)
            print(action)
            print()

        # 计算duration,他的负数为reward，后面可以在改一下
        if line_json['Event'] == 'SparkListenerApplicationStart':
            start_timestamp = line_json['Timestamp']
        if line_json['Event'] == 'SparkListenerApplicationEnd':
            end_timestamp = line_json['Timestamp']

        # 计算state，包括Task的Duration，GC Time，Input Size，Shuffle Read，Shuffle Write，
        if line_json['Event'] == 'SparkListenerTaskEnd':
            try:
                gc_times.append(float(line_json['Task Metrics']['JVM GC Time']) / 1000)
                task_duration = int(line_json['Task Info']['Finish Time']) - int(line_json['Task Info']['Launch Time'])
                task_durations.append(float(task_duration) / 1000)
                input_sizes.append(int(line_json['Task Metrics']['Input Metrics']['Bytes Read']))
                records.append(int(line_json['Task Metrics']['Input Metrics']['Records Read']))
                shuffle_reads.append(int(line_json['Task Metrics']['Shuffle Read Metrics']['Remote Bytes Read']) +
                                     int(line_json['Task Metrics']['Shuffle Read Metrics']['Local Bytes Read']))
                shuffle_writes.append(int(line_json['Task Metrics']['Shuffle Write Metrics']['Shuffle Bytes Written']))
            except:
                print('metrics key error')
                # reward = -10000
                continue
                # break

        if line_json['Event'] == 'SparkListenerJobEnd':
            if line_json['Job Result']['Result'] != 'JobSucceeded':
                break

    if start_timestamp and end_timestamp:
        duration = int(end_timestamp) - int(start_timestamp)
        reward = -duration / 1000
    else:
        print(start_timestamp, end_timestamp)
        return -10000, state

    gc_time_median = np.mean(np.array(gc_times))
    task_duration_median = np.mean(np.array(task_durations))
    input_size_median = np.mean(np.array(input_sizes))
    record_median = np.mean(np.array(records))
    shuffle_read_median = np.mean(np.array(shuffle_reads))
    shuffle_write_median = np.mean(np.array(shuffle_writes))

    state = np.array([task_duration_median, gc_time_median, input_size_median, record_median, shuffle_read_median,
                      shuffle_write_median])

    return reward, state


class Environment(gym.Env):

    def __init__(self,workload):
        self.xth = 0
        self.o_low = o_low
        self.o_high = o_high
        self.a_low = a_low
        self.a_high = a_high
        self.action_space = spaces.Box(low=self.a_low, high=self.a_high, dtype=np.int)
        self.observation_space = spaces.Box(low=self.o_low, high=self.o_high, dtype=np.float32)
        self.state = None
        self.default_action = self.a_low
        self.workload = workload

    def step(self, action, f):
        new_action = []
        # action变换
        for i in range(len(action)):
            new_action.append(int(a_low[i] + (float(action[i] + 2) / 4) * (a_high[i] - a_low[i])))
            # action[i] = int(a_low[i] + ((action[i] + 2) / 4) * (a_high[i] - a_low[i]))
            if new_action[i] > a_high[i]:
                new_action[i] = a_high[i]
            if new_action[i] < a_low[i]:
                new_action[i] = a_low[i]
        assert self.action_space.contains(new_action), "%r (%s) invalid" % (new_action, type(new_action))
        f.write("action:" + str(new_action) + "\n")
        # 首先你得按着action去跑
        code, msg = run_bench(self.workload, new_action)
        # state和reward变化
        # 如果code等于0，表示运行成功，用duration的复数表示reward,同时变化state
        if code == 0:
            reward, self.state = get_rs(self.state, new_action)
        else:
            reward = -10000

        return self.state, reward, False, {}

    def reset(self):
        self.state = self.observation_space.sample()
        # self.state = np.random.rand(4) * self.o_high
        self.counts = 0
        return self.state

    def render(self, mode='human'):
        return None

    def close(self):
        return None


if __name__ == '__main__':
    env = Environment()
    env.reset()
    env.step(env.action_space.sample())
    print(env.state)
    env.step(env.action_space.sample())
    print(env.state)

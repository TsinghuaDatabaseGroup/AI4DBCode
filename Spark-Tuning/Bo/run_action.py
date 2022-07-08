""" 
@author: jdfeng
@contact: jdefeng@stu.xmu.edu.cn
@software: PyCharm 
@file: main.py 
@create: 2020/10/9 17:52 
"""
import shell_content
import os
import time
import signal
import subprocess
import numpy as np
import pydoop.hdfs as hdfs
import json

spark_conf_names = ['spark.default.parallelism', 'spark.driver.cores', 'spark.driver.memory',
                    'spark.driver.maxResultSize',
                    'spark.executor.instances', 'spark.executor.cores', 'spark.executor.memory',
                    'spark.executor.memoryOverhead',
                    'spark.files.maxPartitionBytes', 'spark.memory.fraction', 'spark.memory.storageFraction',
                    'spark.reducer.maxSizeInFlight',
                    'spark.shuffle.compress', 'spark.shuffle.file.buffer', 'spark.shuffle.spill.compress']

AppName = ['ConnectedComponent', 'DecisionTree', 'KMeans', 'LabelPropagation', 'LinearRegression',
           'LogisticRegression', 'PageRank',
           'PCA', 'PregelOperation', 'ShortestPaths', 'StronglyConnectedComponent', 'SVDPlusPlus', 'SVM', 'Terasort',
           'TriangleCount']

result_sizes = ['200m', '500m', '1g', '2g', '4g']

last_log = ""

def get_conf_str(params):
    #[1, 5, 3, 8, 7, 4, 6, 3, 6, 2, 5, 3, 1, 1, 1]
    executor_cores, executor_num, mem_fraction, executor_mem, parallelism_vals, driver_cores,\
    driver_mem, driver_maxResultSize, executor_memoryOverhead, files_maxPartitionBytes, mem_storageFraction,\
        reducer_maxSizeInFlight, shuffle_compress, shuffle_file_buffer, shuffle_spill_compress = params
    return_str = ""
    return_str += ''' --conf "spark.executor.cores=''' + str(executor_cores) + '''" '''
    return_str += ''' --conf "spark.executor.memory=''' + str(executor_mem) + '''g" '''
    return_str += ''' --conf "spark.executor.instances=''' + str(executor_num) + '''" '''
    return_str += ''' --conf "spark.memory.fraction=''' + str(float(mem_fraction)/10) + '''" '''
    return_str += ('''--conf "spark.default.parallelism=''' + str(parallelism_vals) + '''" ''')
    return_str += ('''--conf "spark.driver.cores=''' + str(driver_cores) + '''" ''')
    return_str += ('''--conf "spark.driver.memory=''' + str(driver_mem) + '''g" ''')
    return_str += ('''--conf "spark.driver.maxResultSize=''' + str(result_sizes[int(driver_maxResultSize)]) + '''" ''')
    return_str += ('''--conf "spark.executor.memoryOverhead=''' + str(executor_memoryOverhead * 128) + '''m" ''')
    return_str += ('''--conf "spark.files.maxPartitionBytes=''' + str(files_maxPartitionBytes * 64) + '''m" ''')
    return_str += ('''--conf "spark.memory.storageFraction=''' + str(float(mem_storageFraction)/10) + '''" ''')
    return_str += ('''--conf "spark.reducer.maxSizeInFlight=''' + str(reducer_maxSizeInFlight*32) + '''m" ''')
    bool_vals = ['true', 'false']
    return_str += ('''--conf "spark.shuffle.compress=''' + str(bool_vals[int(shuffle_compress)]) + '''" ''')
    return_str += ('''--conf "spark.shuffle.file.buffer=''' + str(shuffle_file_buffer*32) + '''k" ''')
    return_str += ('''--conf "spark.shuffle.spill.compress=''' + str(bool_vals[int(shuffle_spill_compress)]) + '''" ''')
    return return_str

def get_shell_file(shell_file_path, params, workload_num):
    # 不同workload的sh文件不同
    shell_file = open(shell_file_path, 'w', encoding='utf-8')
    shell_file.write(shell_content.front[workload_num])
    shell_file.write(
        '''    echo_and_run sh -c " ${SPARK_HOME}/bin/spark-submit --class $CLASS \
        --master ${APP_MASTER} ${SPARK_RUN_OPT} ''' +
        get_conf_str(params) + ''' $JAR ${OPTION} 2>&1|tee ${BENCH_NUM}/${APP}_run_${START_TS}.dat"''')
    shell_file.write(shell_content.rear)

def run_command(cmd_string, timeout=100):
    p = subprocess.Popen(cmd_string, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True, close_fds=True, start_new_session=True)
    try:
        (msg, errs) = p.communicate(timeout=timeout)
        ret_code = p.poll()
        if ret_code:
            code = 1
            msg = "[Error]Called Error ： " + str(msg.decode('utf-8'))
        else:
            code = 0
            # msg = str(msg.decode('utf-8'))
            msg = "finished fucking successfully"
    except subprocess.TimeoutExpired:
        os.system("for i in  `yarn application  -list  | awk '{print $1}' | grep application_`; do yarn  application -kill $i; hadoop fs -rm /history/${i}_1; done ")
        p.kill()
        p.terminate()
        os.killpg(p.pid, signal.SIGTERM)
        code = 1
        msg = "[ERROR]Timeout Error : Command '" + cmd_string + "' timed out after " + str(timeout) + " seconds"
    except Exception as e:
        code = 1
        msg = "[ERROR]Unknown Error : " + str(e)
    return code, msg


def get_rs(action):

    # 找到这次的日志文件
    his_file_list = hdfs.ls("/history/")
    log_path = his_file_list[-1]
    global last_log

    #如果两者相等，说明新的并没有生成日志文件，说明报错了
    if last_log == log_path:
        return 10000

    last_log = log_path
    print("Application:" + last_log)
    log_file = hdfs.open(log_path, 'rt', encoding='utf-8')
    # 处理一条历史记录
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

        if line_json['Event'] == 'SparkListenerJobEnd':
            if line_json['Job Result']['Result'] != 'JobSucceeded':
                break

    if start_timestamp and end_timestamp:
        duration = float(int(end_timestamp) - int(start_timestamp))/1000
    else:
        return 10000

    return duration


def run_bench(workload, params):
    workload_num = AppName.index(workload)
    # spark_bench_path = "C:/Users/86159/PycharmProjects/Reinforcement_Learning_in_Action-master/"
    spark_bench_path = "/home/spark_user/lib/spark-bench-legacy/"
    shell_file_path = spark_bench_path + workload + "/bin/my-run.sh"
    get_shell_file(shell_file_path, params, workload_num)
    code, msg = run_command("bash " + shell_file_path, timeout=2000)
    time.sleep(10)
    return code, msg

if __name__ == "__main__":
    ww = "SVDPlusPlus"
    workload_num = AppName.index(ww)
    # spark_bench_path = "C:/Users/86159/PycharmProjects/Reinforcement_Learning_in_Action-master/"
    spark_bench_path = "/home/spark_user/lib/spark-bench-legacy/"
    shell_file_path = spark_bench_path + ww + "/bin/my-run.sh"
    get_shell_file(shell_file_path, [2, 5, 5, 5, 2, 3, 5, 2, 8, 3, 2, 2, 1, 6, 1], workload_num)
    os.system("bash " + shell_file_path)
    # run_bench("PageRank", [1, 5, 3, 8, 7, 4, 6, 3, 6, 2, 5, 3, 1, 1, 1])
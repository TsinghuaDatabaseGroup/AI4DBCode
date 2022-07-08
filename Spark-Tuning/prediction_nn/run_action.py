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
    executor_cores, executor_num, mem_fraction, executor_mem, parallelism_vals, driver_cores,\
    driver_mem, driver_maxResultSize, executor_memoryOverhead, files_maxPartitionBytes, mem_storageFraction,\
        reducer_maxSizeInFlight, shuffle_compress, shuffle_file_buffer, shuffle_spill_compress = params

    # executor_memoryOverhead = 16 - executor_mem
    return_str = ""
    return_str += ''' --conf "spark.executor.cores=''' + str(executor_cores) + '''" '''
    return_str += ''' --conf "spark.executor.memory=''' + str(executor_mem) + '''g" '''
    return_str += ''' --conf "spark.executor.instances=''' + str(executor_num) + '''" '''
    return_str += ''' --conf "spark.memory.fraction=''' + str(float(mem_fraction)/10) + '''" '''
    return_str += ('''--conf "spark.default.parallelism=''' + str(parallelism_vals) + '''" ''')
    return_str += ('''--conf "spark.driver.cores=''' + str(driver_cores) + '''" ''')
    return_str += ('''--conf "spark.driver.memory=''' + str(driver_mem) + '''g" ''')
    return_str += ('''--conf "spark.driver.maxResultSize=''' + str(result_sizes[int(driver_maxResultSize)]) + '''" ''')
    return_str += ('''--conf "spark.executor.memoryOverhead=''' + str(executor_memoryOverhead * 512) + '''m" ''')
    return_str += ('''--conf "spark.files.maxPartitionBytes=''' + str(files_maxPartitionBytes * 64) + '''m" ''')
    return_str += ('''--conf "spark.memory.storageFraction=''' + str(float(mem_storageFraction)/10) + '''" ''')
    return_str += ('''--conf "spark.reducer.maxSizeInFlight=''' + str(reducer_maxSizeInFlight*32) + '''m" ''')
    bool_vals = ['true', 'false']
    return_str += ('''--conf "spark.shuffle.compress=''' + str(bool_vals[int(shuffle_compress)]) + '''" ''')
    return_str += ('''--conf "spark.shuffle.file.buffer=''' + str(shuffle_file_buffer*32) + '''k" ''')
    return_str += ('''--conf "spark.shuffle.spill.compress=''' + str(bool_vals[int(shuffle_spill_compress)]) + '''" ''')
    return_str += ('''--conf "spark.network.timeout=''' + str(300) + '''" ''')
    return return_str

def get_shell_file(shell_file_path, params, workload_num):
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
        os.system("for i in  `yarn application  -list  | awk '{print $1}' | grep application_`; do yarn  application -kill $i; done ")
        p.kill()
        p.terminate()
        os.killpg(p.pid, signal.SIGTERM)
        code = 1
        msg = "[ERROR]Timeout Error : Command '" + cmd_string + "' timed out after " + str(timeout) + " seconds"
    except Exception as e:
        code = 1
        msg = "[ERROR]Unknown Error : " + str(e)
    return code, msg

def read_log(workload):
    path = "log/" + workload
    log = open(path,'r',encoding='utf-8')
    all_stage_info = {}
    for line in log:
        try:
            line_json = json.loads(line)
        except:
            print('json错误')
            continue
        # shuffle read/write、input/output
        if line_json['Event'] == 'SparkListenerTaskEnd':
            cur_stage = line_json['Stage ID']
            # new stage
            if line_json['Stage ID'] not in all_stage_info:
                all_stage_info[cur_stage] = [0, 0, 0, 0]
            # if line_json['Stage ID'] != cur_stage:
            #     cur_metrics, cur_stage = {'input': 0, 'output': 0, 'read': 0, 'write': 0}, line_json['Stage ID']
            try:
                all_stage_info[cur_stage][0] += line_json['Task Metrics']['Input Metrics']['Bytes Read']
                all_stage_info[cur_stage][1] += line_json['Task Metrics']['Output Metrics']['Bytes Written']
                all_stage_info[cur_stage][2] += (line_json['Task Metrics']['Shuffle Read Metrics']['Remote Bytes Read'] +
                                        line_json['Task Metrics']['Shuffle Read Metrics']['Local Bytes Read'])
                all_stage_info[cur_stage][3] += line_json['Task Metrics']['Shuffle Write Metrics']['Shuffle Bytes Written']
            except:
                print('metrics key error')
                break
    return len(all_stage_info.values()), list(all_stage_info.values())

def get_rs(action):

    # find log
    his_file_list = hdfs.ls("/history/")
    log_path = his_file_list[-1]
    global last_log

    if last_log == log_path:
        return 10000

    last_log = log_path
    print("Application:" + last_log)
    log_file = hdfs.open(log_path, 'rt', encoding='utf-8')

    start_timestamp = None
    end_timestamp = None
    for line in log_file:
        try:
            line_json = json.loads(line)
        except:
            print('json错误')
            continue


        if line_json['Event'] == 'SparkListenerEnvironmentUpdate':
            spark_props = line_json['Spark Properties']
            s = ''
            for conf_name in spark_conf_names:
                s = s + spark_props[conf_name] + ", "

            print()
            print(s)
            print(action)
            print()


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
    # os.system("bash " + shell_file_path)
    code, msg = run_command("bash " + shell_file_path, timeout=600)
    time.sleep(10)
    # return code, msg

if __name__ == "__main__":
    run_bench("LinearRegression",[1, 1, 1, 1, 1, 1, 1, 0, 4, 1, 1, 1, 0, 1, 0])
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


AppName = ['ConnectedComponent', 'DecisionTree', 'KMeans', 'LabelPropagation', 'LinearRegression',
           'LogisticRegression', 'PageRank',
           'PCA', 'PregelOperation', 'ShortestPaths', 'StronglyConnectedComponent', 'SVDPlusPlus', 'SVM', 'Terasort',
           'TriangleCount']

result_sizes = ['200m', '500m', '1g', '2g', '4g']

def get_conf_str(params):
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

def run_bench(workload, params):
    workload_num = AppName.index(workload)
    # spark_bench_path = "C:/Users/86159/PycharmProjects/Reinforcement_Learning_in_Action-master/"
    spark_bench_path = "/home/spark_user/lib/spark-bench-legacy/"
    shell_file_path = spark_bench_path + workload + "/bin/my-run.sh"
    get_shell_file(shell_file_path, params, workload_num)
    code, msg = run_command("bash " + shell_file_path, timeout=3600)
    # os.system("bash " + shell_file_path)
    time.sleep(10)
    # return 1,2
    return code, msg

if __name__ == "__main__":
    run_bench("LinearRegression",[1, 1, 1, 1, 1, 1, 1, 0, 4, 1, 1, 1, 0, 1, 0])
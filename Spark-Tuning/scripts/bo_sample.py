from skopt import gp_minimize
import shell_content
import os
import time
import signal
import subprocess

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="run workload with bayes_optimation")

    parser.add_argument('spark_bench_path', type=str, help='path to sparkbench')

    return parser.parse_args()

args = parse_args()




result_sizes = ['200m', '500m', '1g', '2g', '4g']
workload = 0
times = 500
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
    return_str += ('''--conf "spark.driver.maxResultSize=''' + str(result_sizes[driver_maxResultSize]) + '''" ''')
    return_str += ('''--conf "spark.executor.memoryOverhead=''' + str(executor_memoryOverhead * 128) + '''m" ''')
    return_str += ('''--conf "spark.files.maxPartitionBytes=''' + str(files_maxPartitionBytes * 64) + '''m" ''')
    return_str += ('''--conf "spark.memory.storageFraction=''' + str(float(mem_storageFraction)/10) + '''" ''')
    return_str += ('''--conf "spark.reducer.maxSizeInFlight=''' + str(reducer_maxSizeInFlight*32) + '''m" ''')
    bool_vals = ['true', 'false']
    return_str += ('''--conf "spark.shuffle.compress=''' + str(bool_vals[shuffle_compress]) + '''" ''')
    return_str += ('''--conf "spark.shuffle.file.buffer=''' + str(shuffle_file_buffer*32) + '''k" ''')
    return_str += ('''--conf "spark.shuffle.spill.compress=''' + str(bool_vals[shuffle_spill_compress]) + '''" ''')
    return return_str


def get_shell_file(shell_file_path, params):
    global workload
    shell_file = open(shell_file_path, 'w', encoding='utf-8')
    shell_file.write(shell_content.front[workload])
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
        print(os.popen("yarn application  -list  | awk '{print $1}' | grep application_", ).read())
        if ret_code:
            code = 1
            msg = "[Error]Called Error ： " + str(msg.decode('utf-8'))
        else:
            code = 0
            # msg = str(msg.decode('utf-8'))
            msg = "finished fucking successfully"
    except subprocess.TimeoutExpired:
        os.system("for i in  `yarn application  -list  | awk '{print $1}' | grep application_`; do yarn  application -kill $i; hadoop fs -rm /history/${i}_1.inprogress; done ")
        p.kill()
        p.terminate()
        os.killpg(p.pid, signal.SIGTERM)
        code = 1
        msg = "[ERROR]Timeout Error : Command '" + cmd_string + "' timed out after " + str(timeout) + " seconds"
    except Exception as e:
        code = 1
        msg = "[ERROR]Unknown Error : " + str(e)
    return code, msg

def get_duration_from_sh(sample):
    global times
    spark_bench_path = args.spark_bench_path
    AppName = ['ConnectedComponent', 'DecisionTree', 'KMeans', 'LabelPropagation', 'LinearRegression',
                'LogisticRegression', 'PageRank',
                'PCA', 'PregelOperation', 'ShortestPaths', 'StronglyConnectedComponent', 'SVM', 'Terasort',
                'TriangleCount']
    shell_file_path = spark_bench_path + AppName[workload] + '/bin/my-run.sh'
    get_shell_file(shell_file_path, sample)
    start = time.time()
    # return_code = os.system("bash " + spark_bench_path + shell_file_path[workload])
    code, msg = run_command("bash " + shell_file_path, timeout=7200)

    end = time.time()
    print("\n" + msg)
    print('total time: ' + str(end - start))
    if code == 0:
        times -= 1
        print('\nNow is the ' + str(workload) + 'th workload ' + AppName[workload] + '\n' + str(
            times) + ' times left\n\n')
        time.sleep(1)
        return end - start
    else:
        return 300


def bayes_sample():
    global X
    global Y
    global workload
    global times
    # 1executor_cores, 2executor_instances, 3mem_fraction, 4executor_mem, 5parallelism,6driver.cores, 7driver.memory, 8driver.maxResultSize, 9executor.memoryOverhead
    # files.maxPartitionBytes, memory.storageFraction, reducer.maxSizeInFlight, shuffle.compress, file.buffer,shuffle.spill.compress
    dimension = [(1, 8), (1, 8), (1, 9), (1, 8), (1, 8), (1, 8), (1, 8), (0, 4), (4, 8), (1, 4), (1, 9),
                 (1, 4), (0, 1), (1, 8), (0, 1)]

    w = [i for i in range(14)]
    for workload_count in w:
        workload = workload_count
        times = 500
        res = gp_minimize(get_duration_from_sh,
                          # the function to minimize, get_duration_from_csv or get_duration_from_sh
                          dimension,  # the bounds on each dimension of x
                          acq_func="PI",  # the acquisition function
                          n_calls=500,  # the number of evaluations of f
                          n_random_starts=100,  # the number of random initialization points
                          noise=0.1 ** 2,  # the noise level (optional)
                          )



if __name__ == '__main__':
    bayes_sample()

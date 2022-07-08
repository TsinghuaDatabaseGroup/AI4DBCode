from gp_minimize_new import gp_minimize
import pandas as pd
import random
import run_action

random.seed(2021)
# workloads = ['ConnectedComponent', 'PageRank', 'ShortestPaths', 'StronglyConnectedComponent',
#              'PregelOperation', 'LabelPropagation', 'TriangleCount', 'SVDPlusPlus']
# app_names = ['Spark ConnectedComponent Application', 'Spark PageRank Application',
#              'Spark ShortestPath Application','Spark StronglyConnectedComponent Application',
#              'Spark PregelOperation Application', 'Spark LabelPropagation Application', 'Spark TriangleCount Application', 'Spark SVDPlusPlus Application']
workloads = ['PCA', 'SVDPlusPlus', 'ConnectedComponent',
             'PregelOperation', 'TriangleCount']
app_names = ['Spark PCA Example', 'Spark SVDPlusPlus Application', 'Spark ConnectedComponent Application',
             'Spark PregelOperation Application', 'Spark TriangleCount Application']

dataset_path = 'data/dataset_test_8_10000M_1.csv'

df_all = pd.read_csv(dataset_path, sep=',', low_memory=False)

df_alls = df_all.groupby(["AppName"])

result_sizes = {0.2: 0,
                0.5: 1,
                1: 2,
                2: 3,
                4: 4, }

def get_duration_from_sh(sample):
    # 先跑一遍
    code, msg = run_action.run_bench(workload, sample)
    # 算时间
    if code == 0:
        duration = run_action.get_rs(sample)
    else:
        duration = 10000
    # duration = np.sum(np.array(sample))
    return duration


def bayes_sample():
    dimension = [(1, 8), (1, 8), (1, 9), (1, 8), (1, 8),
                 (1, 8), (1, 8), (0, 4), (4, 8), (1, 4),
                 (1, 9), (1, 4), (0, 1), (1, 8), (0, 1)]

    #起始点构造
    init_samples = []
    init_durations = {}
    # executor_cores, executor_num, mem_fraction, executor_mem, parallelism_vals, driver_cores,\
    #     driver_mem, driver_maxResultSize, executor_memoryOverhead, files_maxPartitionBytes, mem_storageFraction,\
    #         reducer_maxSizeInFlight, shuffle_compress, shuffle_file_buffer, shuffle_spill_compress = params 按这个顺序
    for w_name, w_df in df_alls:
        if w_name == app_name:
            for w in w_df.values:
                init_sample = [w[8], w[7], w[12] * 10, w[9], w[3], w[4], w[5], result_sizes[w[6]], float(w[10]) / 128,
                               w[11] / 64,
                               w[13] * 10, w[14] / 32, w[15],
                               w[16] / 32, w[17], w[2]]
                init_duration = w[2]
                init_samples.append(init_sample)
                init_durations[str(init_sample)] = init_duration

    #有起始点情况
    if init_samples:
        res = gp_minimize(get_duration_from_sh,
                          # the function to minimize, get_duration_from_csv or get_duration_from_sh
                          dimension,  # the bounds on each dimension of x
                          acq_func="EI",  # the acquisition function
                          n_calls=200,  # the number of evaluations of f
                          n_random_starts=len(init_samples),  # the number of random initialization points
                          noise=2 ** 2,  # the noise level (optional)
                          init_samples=init_samples,
                          init_durations=init_durations
                          )
    #没有起始点情况
    else:
        res = gp_minimize(get_duration_from_sh,
                          # the function to minimize, get_duration_from_csv or get_duration_from_sh
                          dimension,  # the bounds on each dimension of x
                          acq_func="EI",  # the acquisition function
                          n_calls=100,  # the number of evaluations of f
                          n_random_starts=5,  # the number of random initialization points
                          noise=2 ** 2,  # the noise level (optional)
                          init_samples=init_samples,
                          init_durations=init_durations
                          )

if __name__ == '__main__':
    # svr_regression  bayesian_ridge_regression  linear_regression  sgd_regression
    # lasso_regression  gbr_regression  mlp_regression
    for i in range(0, len(workloads)):
        workload = workloads[i]
        app_name = app_names[i]
        f = open('result.txt', 'a', encoding='utf-8')
        f.write(workload + ":\n")
        print(workload + ":")
        f.close()
        bayes_sample()

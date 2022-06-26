import os
import psycopg2
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import base

# 统计重复率：

def cal_duplicate_rate(dbname):
    root_path = os.path.abspath('.') + '/' + dbname
    query_file_list = os.listdir(root_path)
    queries = []
    for query_file in query_file_list:
        path = root_path + '/' + query_file
        with open(path, 'r') as f:
            queries.extend(f.read().split(';'))
    total_query_count = len(queries)
    uniq_query_count = len(set(queries))
    duplicate_rate = (total_query_count - uniq_query_count) / total_query_count
    with open(root_path + '/duplicate_rate', 'w') as f:
        f.write(str(duplicate_rate))


# 统计执行率
def cal_can_execute_rate(dbname):
    root_path = os.path.abspath('.') + '/' + dbname
    metric_path = root_path + '/' + '{0}_metric'.format(dbname)
    query_cost = pd.read_csv(metric_path, index_col=0)
    # 统计执行率
    total_query_count = query_cost.shape[0]
    can_execute_query_count = query_cost.loc[query_cost['can_execute'] == True].shape[0]
    execute_rate = can_execute_query_count / total_query_count
    print('can execute rate:', execute_rate)
    with open(root_path + '/execute_rate', 'w') as f:
        f.write(str(execute_rate))


# 结果可视化
def show_cost_distribute(dbname):
    root_path = os.path.abspath('.') + '/' + dbname
    metric_path = root_path + '/' + '{0}_metric'.format(dbname)
    query_cost = pd.read_csv(metric_path, index_col=0)
    # 可视化分布
    fig = plt.figure()
    plt.style.use('ggplot')
    plt.title('cost distribution')
    plt.xlabel('query_id')
    plt.ylabel('cost')
    plt.grid(True)

    plt.scatter(np.arange(0, query_cost.shape[0]), query_cost.cost.tolist(), label=dbname, s=1)
    plt.ylim((-1, 100000000))
    plt.locator_params('y', nbins=20)
    plt.show()
    fig.savefig(root_path + '/cost_distribution.png')


def cal_statistic(dbname):
    root_path = os.path.abspath('.') + '/' + dbname
    metric_path = root_path + '/' + '{0}_metric'.format(dbname)
    query_cost = pd.read_csv(metric_path, index_col=0)
    can_execute_query = query_cost.loc[query_cost['can_execute'] == True]
    can_execute_query_count = can_execute_query.shape[0]
    query_cost_equal_zero_count = can_execute_query.loc[can_execute_query['cost'] == 0].shape[0]
    print('cost为0的比例', query_cost_equal_zero_count / can_execute_query_count)
    query_cost_less_10 = can_execute_query.loc[can_execute_query['cost'] < 10].shape[0]
    print('cost小于10', query_cost_less_10 / can_execute_query_count)
    query_cost_less_100 = can_execute_query.loc[can_execute_query['cost'] < 100].shape[0]
    print('cost小于100', query_cost_less_100 / can_execute_query_count)
    query_cost_less_1000 = can_execute_query.loc[can_execute_query['cost'] < 1000].shape[0]
    print('cost小于1000', query_cost_less_1000 / can_execute_query_count)

# cal_cost('tpch')
# cal_cost('imdbload')
# cal_cost('xuetang')
# cal_duplicate_rate('tpch')
# cal_duplicate_rate('imdbload')
# cal_duplicate_rate('xuetang')
# cal_can_execute_rate('tpch)
# cal_can_execute_rate('imdbload')
# cal_can_execute_rate('xuetang')

# show_cost_distribute('tpch')
# show_cost_distribute('imdbload')
# show_cost_distribute('xuetang')


# cal_file_e_info('tpch', '/home/lixizhang/learnSQL/sqlsmith/tpch/tpch10000_0')
# cal_file_r_info('tpch', '/home/lixizhang/learnSQL/sqlsmith/tpch/tpch10000_0')

# cal_file_e_info('imdbload', '/home/lixizhang/learnSQL/sqlsmith/imdbload/imdbload10000_0')
# cal_file_r_info('imdbload', '/home/lixizhang/learnSQL/sqlsmith/imdbload/imdbload10000_0')

# cal_statistic('imdbload')
# cal_statistic('xuetang')

# print(cal_point_accuracy('/home/lixizhang/learnSQL/sqlsmith/tpch/tpch10000_0', 0, 10, 'card'))
# print(cal_range_accuracy('/home/lixizhang/learnSQL/sqlsmith/tpch/tpch10000_0', (1000, 2000), 'cost'))

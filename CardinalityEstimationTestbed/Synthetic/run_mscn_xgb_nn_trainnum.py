import argparse
import os

parser = argparse.ArgumentParser(description='mscn_xgb_nn')

parser.add_argument('--version', type=str, help='datasets_dir', default='cols_4_distinct_1000_corr_5_skew_5')
args = parser.parse_args()
version = args.version
'''
# sql1
pretrain = 'python preprocessing.py --datasets-dir /home/zhangjintao/Benchmark3/sql-modelsize/sql/1/ --raw-query-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/1/' + version + \
    'train.sql' + ' --min-max-file /home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master/data/' + version + '_min_max_vals.csv ' + '--table ' + version + ' --alias cdcs'
pretest = 'python preprocessing.py --datasets-dir /home/zhangjintao/Benchmark3/sql-modelsize/sql/1/ --raw-query-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/1/' + version + \
    'test.sql' + ' --min-max-file /home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master/data/' + version + '_min_max_vals.csv ' + '--table ' + version + ' --alias cdcs'
train = 'python train.py --min-max-file /home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master/data/' + version + \
    '_min_max_vals.csv --queries 2500 --epochs 100 --batch 1024 --hid 256 --train-query-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/1/' + version + \
    'train.sql --test-query-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/1/' + version + 'test.sql --train --version ' + version
test = 'python train.py --min-max-file /home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master/data/' + version + \
    '_min_max_vals.csv --queries 2500 --epochs 100 --batch 1024 --hid 256 --train-query-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/1/' + version + \
    'train.sql --test-query-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/1/' + version + 'test.sql --version ' + version


os.chdir('/home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master/mscn')
os.system(pretrain)
os.system(pretest)
os.chdir('/home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master')
os.system(train)
os.system(test)

# sql2
pretrain = 'python preprocessing.py --datasets-dir /home/zhangjintao/Benchmark3/sql-modelsize/sql/2/ --raw-query-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/2/' + version + \
    'train.sql' + ' --min-max-file /home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master/data/' + version + '_min_max_vals.csv ' + '--table ' + version + ' --alias cdcs'
pretest = 'python preprocessing.py --datasets-dir /home/zhangjintao/Benchmark3/sql-modelsize/sql/2/ --raw-query-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/2/' + version + \
    'test.sql' + ' --min-max-file /home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master/data/' + version + '_min_max_vals.csv ' + '--table ' + version + ' --alias cdcs'
train = 'python train.py --min-max-file /home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master/data/' + version + \
    '_min_max_vals.csv --queries 5000 --epochs 100 --batch 1024 --hid 256 --train-query-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/2/' + version + \
    'train.sql --test-query-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/2/' + version + 'test.sql --train --version ' + version
test = 'python train.py --min-max-file /home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master/data/' + version + \
    '_min_max_vals.csv --queries 5000 --epochs 100 --batch 1024 --hid 256 --train-query-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/2/' + version + \
    'train.sql --test-query-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/2/' + version + 'test.sql --version ' + version


os.chdir('/home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master/mscn')
os.system(pretrain)
os.system(pretest)
os.chdir('/home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master')
os.system(train)
os.system(test)
'''
# sql3
pretrain = 'python preprocessing.py --datasets-dir /home/zhangjintao/Benchmark3/sql-modelsize/sql/3/ --raw-query-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/3/' + version + \
           'train.sql' + ' --min-max-file /home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master/data/' + version + '_min_max_vals.csv ' + '--table ' + version + ' --alias cdcs'
pretest = 'python preprocessing.py --datasets-dir /home/zhangjintao/Benchmark3/sql-modelsize/sql/3/ --raw-query-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/3/' + version + \
          'test.sql' + ' --min-max-file /home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master/data/' + version + '_min_max_vals.csv ' + '--table ' + version + ' --alias cdcs'
train = 'python train.py --min-max-file /home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master/data/' + version + \
        '_min_max_vals.csv --queries 7500 --epochs 100 --batch 1024 --hid 256 --train-query-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/3/' + version + \
        'train.sql --test-query-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/3/' + version + 'test.sql --train --version ' + version
test = 'python train.py --min-max-file /home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master/data/' + version + \
       '_min_max_vals.csv --queries 7500 --epochs 100 --batch 1024 --hid 256 --train-query-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/3/' + version + \
       'train.sql --test-query-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/3/' + version + 'test.sql --version ' + version

os.chdir('/home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master/mscn')
os.system(pretrain)
os.system(pretest)
os.chdir('/home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master')
os.system(train)
os.system(test)

# parser.add_argument('--model', type=str, help='nn||xgb', default='nn')

# model = args.model

'''
# sql1
run = 'python run.py --train-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/1/' + version + 'train.sql' + ' --test-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/1/' + version + 'test.sql' + \
    ' --min-max-file /home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master/data/' + version + '_min_max_vals.csv ' + '--model ' + 'nn' + ' --version ' + version

os.chdir('/home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/xgboost')
os.system(run)

run = 'python run.py --train-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/1/' + version + 'train.sql' + ' --test-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/1/' + version + 'test.sql' + \
    ' --min-max-file /home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master/data/' + version + '_min_max_vals.csv ' + '--model ' + 'xgb' + ' --version ' + version

os.chdir('/home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/xgboost')
os.system(run)
# sql2
run = 'python run.py --train-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/2/' + version + 'train.sql' + ' --test-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/2/' + version + 'test.sql' + \
    ' --min-max-file /home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master/data/' + version + '_min_max_vals.csv ' + '--model ' + 'nn' + ' --version ' + version

os.chdir('/home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/xgboost')
os.system(run)

run = 'python run.py --train-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/2/' + version + 'train.sql' + ' --test-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/2/' + version + 'test.sql' + \
    ' --min-max-file /home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master/data/' + version + '_min_max_vals.csv ' + '--model ' + 'xgb' + ' --version ' + version

os.chdir('/home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/xgboost')
os.system(run)
'''
# sql3
run = 'python run.py --train-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/3/' + version + 'train.sql' + ' --test-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/3/' + version + 'test.sql' + \
      ' --min-max-file /home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master/data/' + version + '_min_max_vals.csv ' + '--model ' + 'nn' + ' --version ' + version

os.chdir('/home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/xgboost')
os.system(run)

run = 'python run.py --train-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/3/' + version + 'train.sql' + ' --test-file /home/zhangjintao/Benchmark3/sql-modelsize/sql/3/' + version + 'test.sql' + \
      ' --min-max-file /home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master/data/' + version + '_min_max_vals.csv ' + '--model ' + 'xgb' + ' --version ' + version

os.chdir('/home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/xgboost')
os.system(run)

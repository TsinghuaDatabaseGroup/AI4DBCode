import argparse
import os

parser = argparse.ArgumentParser(description='mscn')

parser.add_argument('--version', type=str, help='datasets_dir', default='cols_4_distinct_1000_corr_5_skew_5')
args = parser.parse_args()
version = args.version

pretrain = 'python preprocessing.py --datasets-dir ../../csvdata_sql --raw-query-file ../../sql_truecard/' + version + \
           'train.sql' + ' --min-max-file ../data/' + version + '_min_max_vals.csv ' + '--table ' + version + ' --alias cdcs'
pretest = 'python preprocessing.py --datasets-dir ../../csvdata_sql --raw-query-file ../../sql_truecard/' + version + \
          'test.sql' + ' --min-max-file ../data/' + version + '_min_max_vals.csv ' + '--table ' + version + ' --alias cdcs'
train = 'python train.py --min-max-file ./data/' + version + \
        '_min_max_vals.csv --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../sql_truecard/' + version + \
        'train.sql --test-query-file ../sql_truecard/' + version + 'test.sql --train --version ' + version
test = 'python train.py --min-max-file ./data/' + version + \
       '_min_max_vals.csv --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../sql_truecard/' + version + \
       'train.sql --test-query-file ../sql_truecard/' + version + 'test.sql --version ' + version

os.chdir('./learnedcardinalities-master/mscn')
os.system(pretrain)
os.system(pretest)
os.chdir('..')
os.system(train)
os.system(test)

'''
cd mscn

python preprocessing.py --datasets-dir ./csvdata_sql/ --raw-query-file ./sql_truecard/train-num.sql --min-max-file /home/jintao/CardinalityEstimationBenchmark/learnedcardinalities-master/data/distinct3_min_max_vals.csv --alias distinct3

python preprocessing.py --datasets-dir /home/jintao/CardinalityEstimationBenchmark/Corrskewdis/distinct3 --raw-query-file /home/jintao/CardinalityEstimationBenchmark/train-test-data/distinct-sql/distinct3/test-num.sql --min-max-file /home/jintao/CardinalityEstimationBenchmark/learnedcardinalities-master/data/distinct3_min_max_vals.csv --alias distinct3

cd ..

python train.py --min-max-file /home/jintao/CardinalityEstimationBenchmark/learnedcardinalities-master/data/distinct3_min_max_vals.csv --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file /home/jintao/CardinalityEstimationBenchmark/train-test-data/distinct-sql/distinct3/train-num.sql --test-query-file /home/jintao/CardinalityEstimationBenchmark/train-test-data/distinct-sql/distinct3/test-num.sql --train

python train.py --min-max-file /home/jintao/CardinalityEstimationBenchmark/learnedcardinalities-master/data/distinct3_min_max_vals.csv --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file /home/jintao/CardinalityEstimationBenchmark/train-test-data/distinct-sql/distinct3/train-num.sql --test-query-file /home/jintao/CardinalityEstimationBenchmark/train-test-data/distinct-sql/distinct3/test-num.sql
'''

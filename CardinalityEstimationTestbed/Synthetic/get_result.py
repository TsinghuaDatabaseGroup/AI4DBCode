
import argparse
import os

parser = argparse.ArgumentParser(description='generatedata')

parser.add_argument('--distinct', type=int, help='datasets_dir', default=1000)
parser.add_argument('--cols', type=int, help='datasets_dir', default=4)
parser.add_argument('--corr', type=int, help='datasets_dir', default=5)
parser.add_argument('--skew', type=int, help='datasets_dir', default=5)
parser.add_argument('--method', type=str, help='datasets_dir', default='nn')

args = parser.parse_args()
# os.chdir('/home/zhangjintao/Benchmark3')
# cols_4_distinct_1000_corr_5_skew_5

cols = args.cols
distinct = args.distinct
corr = args.corr
skew = args.skew
method = args.method

version = 'cols_' + str(cols) + '_distinct_' + str(distinct) + '_corr_' + str(corr) + '_skew_' + str(skew)

# mscn
if method == 'mscn':
    os.system('python run_mscn.py --version ' + version)

# xgb
if method == 'xgb':
    os.system('python run_xgb_nn.py --version ' + version + ' --model xgb')

# nn
if method == 'nn':
    os.system('python run_xgb_nn.py --version ' + version + ' --model nn')

# deepdb
if method == 'deepdb':
    os.system('python run_deepdb.py --version ' + version)

# naru
if method == 'naru':
    os.chdir('./naru')
    os.system(
        'python train_model.py --version ' + version + ' --num-gpus=1 --dataset=dmv --epochs=70 --warmups=8000 --bs=2048 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking')
    os.system(
        'python eval_model.py --testfilepath ../sql_truecard/ --version ' + version + ' --table ' + version + ' --alias cdcs --dataset=dmv --glob=\'<ckpt from above>\' --num-queries=1000 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking')
    os.chdir('..')

# bayesian
if method == 'bayesian':
    os.chdir('./bayesian')
    os.system('python3 eval_model.py --dataset=' + version + ' --num-queries=60 --run-bn')
    os.chdir('..')

# kde
if method == 'kde':
    os.chdir('./kde_python')
    os.system('python kde.py --train-file ../sql_truecard/' + version +'train.sql --test-file ../sql_truecard/' + version +'test.sql --min-max-file ../learnedcardinalities-master/data/' + version +'_min_max_vals.csv --version ' + version)
    os.chdir('..')

print('cols_' + str(cols) + '_distinct_' + str(distinct) + '_corr_' + str(corr) + '_skew_' + str(skew) + 'is OK.')

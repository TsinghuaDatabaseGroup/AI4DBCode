import argparse
import os

parser = argparse.ArgumentParser(description='generatedata')
parser.add_argument('--cols', type=int, help='datasets_dir', default=4)
parser.add_argument('--xt', type=int, help='datasets_dir', default=4)
parser.add_argument('--model', type=str, help='nn or xgb', default=nn)
args = parser.parse_args()
cols = args.cols
xt = args.xt
model = args.model

if model == 'nn':
    if cols == 2:
        os.system(
            'python run.py --train-file ../train-test-data/imdb-cols-sql/2/train-2-num.sql --test-file ../train-test-data/imdb-cols-sql/2/test-2-num.sql --min-max-file ../learnedcardinalities-master/data/col2_min_max_vals.csv --model nn')
    elif cols == 4:
        os.system(
            'python run.py --train-file ../train-test-data/imdb-cols-sql/4/train-4-num.sql --test-file ../train-test-data/imdb-cols-sql/4/test-only4-num.sql --min-max-file ../learnedcardinalities-master/data/col4_min_max_vals.csv --model nn')
    elif cols == 6:
        os.system(
            'python run.py --train-file ../train-test-data/imdb-cols-sql/6/train-6-num.sql --test-file ../train-test-data/imdb-cols-sql/6/test-only6-num.sql --min-max-file ../learnedcardinalities-master/data/col6_min_max_vals.csv --model nn')
    elif cols == 8:
        os.system(
            'python run.py --train-file ../train-test-data/imdb-cols-sql/8/train-8-num.sql --test-file ../train-test-data/imdb-cols-sql/8/test-only8-num.sql --min-max-file ../learnedcardinalities-master/data/col8_min_max_vals.csv --model nn')

    elif xt == 2:
        os.system(
            'python run_xtzx.py --train-file ../train-test-data/xtzx-data-sql/2/train-2-num.sql --test-file ../train-test-data/xtzx-data-sql/2/test-2-num.sql --min-max-file ../learnedcardinalities-master/data/xtcol2_min_max_vals.csv --model nn')
    elif xt == 4:
        os.system(
            'python run_xtzx.py --train-file ../train-test-data/xtzx-data-sql/4/train-4-num.sql --test-file ../train-test-data/xtzx-data-sql/4/test-only4-num.sql --min-max-file ../learnedcardinalities-master/data/xtcol4_min_max_vals.csv --model nn')
    elif xt == 6:
        os.system(
            'python run_xtzx.py --train-file ../train-test-data/xtzx-data-sql/6/train-6-num.sql --test-file ../train-test-data/xtzx-data-sql/6/test-only6-num.sql --min-max-file ../learnedcardinalities-master/data/xtcol6_min_max_vals.csv --model nn')
    elif xt == 8:
        os.system(
            'python run_xtzx.py --train-file ../train-test-data/xtzx-data-sql/8/train-8-num.sql --test-file ../train-test-data/xtzx-data-sql/8/test-only8-num.sql --min-max-file ../learnedcardinalities-master/data/xtcol8_min_max_vals.csv --model nn')
    elif xt == 10:
        os.system(
            'python run_xtzx.py --train-file ../train-test-data/xtzx-data-sql/10/train-10-num.sql --test-file ../train-test-data/xtzx-data-sql/10/test-only10-num.sql --min-max-file ../learnedcardinalities-master/data/xtcol10_min_max_vals.csv --model nn')
else:
    if cols == 2:
        os.system(
            'python run.py --train-file ../train-test-data/imdb-cols-sql/2/train-2-num.sql --test-file ../train-test-data/imdb-cols-sql/2/test-2-num.sql --min-max-file ../learnedcardinalities-master/data/col2_min_max_vals.csv --model xgb')
    elif cols == 4:
        os.system(
            'python run.py --train-file ../train-test-data/imdb-cols-sql/4/train-4-num.sql --test-file ../train-test-data/imdb-cols-sql/4/test-only4-num.sql --min-max-file ../learnedcardinalities-master/data/col4_min_max_vals.csv --model xgb')
    elif cols == 6:
        os.system(
            'python run.py --train-file ../train-test-data/imdb-cols-sql/6/train-6-num.sql --test-file ../train-test-data/imdb-cols-sql/6/test-only6-num.sql --min-max-file ../learnedcardinalities-master/data/col6_min_max_vals.csv --model xgb')
    elif cols == 8:
        os.system(
            'python run.py --train-file ../train-test-data/imdb-cols-sql/8/train-8-num.sql --test-file ../train-test-data/imdb-cols-sql/8/test-only8-num.sql --min-max-file ../learnedcardinalities-master/data/col8_min_max_vals.csv --model xgb')

    elif xt == 2:
        os.system(
            'python run_xtzx.py --train-file ../train-test-data/xtzx-data-sql/2/train-2-num.sql --test-file ../train-test-data/xtzx-data-sql/2/test-2-num.sql --min-max-file ../learnedcardinalities-master/data/xtcol2_min_max_vals.csv --model xgb')
    elif xt == 4:
        os.system(
            'python run_xtzx.py --train-file ../train-test-data/xtzx-data-sql/4/train-4-num.sql --test-file ../train-test-data/xtzx-data-sql/4/test-only4-num.sql --min-max-file ../learnedcardinalities-master/data/xtcol4_min_max_vals.csv --model xgb')
    elif xt == 6:
        os.system(
            'python run_xtzx.py --train-file ../train-test-data/xtzx-data-sql/6/train-6-num.sql --test-file ../train-test-data/xtzx-data-sql/6/test-only6-num.sql --min-max-file ../learnedcardinalities-master/data/xtcol6_min_max_vals.csv --model xgb')
    elif xt == 8:
        os.system(
            'python run_xtzx.py --train-file ../train-test-data/xtzx-data-sql/8/train-8-num.sql --test-file ../train-test-data/xtzx-data-sql/8/test-only8-num.sql --min-max-file ../learnedcardinalities-master/data/xtcol8_min_max_vals.csv --model xgb')
    elif xt == 10:
        os.system(
            'python run_xtzx.py --train-file ../train-test-data/xtzx-data-sql/10/train-10-num.sql --test-file ../train-test-data/xtzx-data-sql/10/test-only10-num.sql --min-max-file ../learnedcardinalities-master/data/xtcol10_min_max_vals.csv --model xgb')

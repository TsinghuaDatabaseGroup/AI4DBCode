import argparse
import os

parser = argparse.ArgumentParser(description='generatedata')
parser.add_argument('--cols', type=int, help='datasets_dir', default=4)
parser.add_argument('--xt', type=int, help='datasets_dir', default=4)
args = parser.parse_args()
cols = args.cols
xt = args.xt

os.chdir('./mscn')
os.system(
    'python preprocessing-job.py --datasets-dir ../../train-test-data/imdbdata-num/ --raw-query-file ../../train-test-data/imdb-cols-sql/4/train-4-num.sql --min-max-file ../data/col4_min_max_vals.csv')
os.system(
    'python preprocessing-job.py --datasets-dir ../../train-test-data/imdbdata-num/ --raw-query-file ../../train-test-data/imdb-cols-sql/4/test-only4-num.sql --min-max-file ../data/col4_min_max_vals.csv')

os.system(
    'python preprocessing-job.py --datasets-dir ../../train-test-data/imdbdata-num/ --raw-query-file ../../train-test-data/imdb-cols-sql/2/train-2-num.sql --min-max-file ../data/col2_min_max_vals.csv')
os.system(
    'python preprocessing-job.py --datasets-dir ../../train-test-data/imdbdata-num/ --raw-query-file ../../train-test-data/imdb-cols-sql/2/test-2-num.sql --min-max-file ../data/col2_min_max_vals.csv')

os.system(
    'python preprocessing-job.py --datasets-dir ../../train-test-data/imdbdata-num/ --raw-query-file ../../train-test-data/imdb-cols-sql/6/train-6-num.sql --min-max-file ../data/col6_min_max_vals.csv')
os.system(
    'python preprocessing-job.py --datasets-dir ../../train-test-data/imdbdata-num/ --raw-query-file ../../train-test-data/imdb-cols-sql/6/test-only6-num.sql --min-max-file ../data/col6_min_max_vals.csv')

os.system(
    'python preprocessing-job.py --datasets-dir ../../train-test-data/imdbdata-num/ --raw-query-file ../../train-test-data/imdb-cols-sql/8/train-8-num.sql --min-max-file ../data/col8_min_max_vals.csv')
os.system(
    'python preprocessing-job.py --datasets-dir ../../train-test-data/imdbdata-num/ --raw-query-file ../../train-test-data/imdb-cols-sql/8/test-only8-num.sql --min-max-file ../data/col8_min_max_vals.csv')
os.chdir('..')

os.chdir('./mscn_xtzx')
os.system(
    'python preprocessing-job.py --datasets-dir ../../train-test-data/xtzx-data-sql/ --raw-query-file ../../train-test-data/xtzx-data-sql/4/train-4-num.sql --min-max-file ../data/xtcol4_min_max_vals.csv')
os.system(
    'python preprocessing-job.py --datasets-dir ../../train-test-data/xtzx-data-sql/ --raw-query-file ../../train-test-data/xtzx-data-sql/4/test-only4-num.sql --min-max-file ../data/xtcol4_min_max_vals.csv')

os.system(
    'python preprocessing-job.py --datasets-dir ../../train-test-data/xtzx-data-sql/ --raw-query-file ../../train-test-data/xtzx-data-sql/2/train-2-num.sql --min-max-file ../data/xtcol2_min_max_vals.csv')
os.system(
    'python preprocessing-job.py --datasets-dir ../../train-test-data/xtzx-data-sql/ --raw-query-file ../../train-test-data/xtzx-data-sql/2/test-2-num.sql --min-max-file ../data/xtcol2_min_max_vals.csv')

os.system(
    'python preprocessing-job.py --datasets-dir ../../train-test-data/xtzx-data-sql/ --raw-query-file ../../train-test-data/xtzx-data-sql/6/train-6-num.sql --min-max-file ../data/xtcol6_min_max_vals.csv')
os.system(
    'python preprocessing-job.py --datasets-dir ../../train-test-data/xtzx-data-sql/ --raw-query-file ../../train-test-data/xtzx-data-sql/6/test-only6-num.sql --min-max-file ../data/xtcol6_min_max_vals.csv')

os.system(
    'python preprocessing-job.py --datasets-dir ../../train-test-data/xtzx-data-sql/ --raw-query-file ../../train-test-data/xtzx-data-sql/8/train-8-num.sql --min-max-file ../data/xtcol8_min_max_vals.csv')
os.system(
    'python preprocessing-job.py --datasets-dir ../../train-test-data/xtzx-data-sql/ --raw-query-file ../../train-test-data/xtzx-data-sql/8/test-only8-num.sql --min-max-file ../data/xtcol8_min_max_vals.csv')

os.system(
    'python preprocessing-job.py --datasets-dir ../../train-test-data/xtzx-data-sql/ --raw-query-file ../../train-test-data/xtzx-data-sql/10/train-10-num.sql --min-max-file ../data/xtcol10_min_max_vals.csv')
os.system(
    'python preprocessing-job.py --datasets-dir ../../train-test-data/xtzx-data-sql/ --raw-query-file ../../train-test-data/xtzx-data-sql/10/test-only10-num.sql --min-max-file ../data/xtcol10_min_max_vals.csv')
os.chdir('..')

if cols == 2:
    os.system(
        'python train.py --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../train-test-data/imdb-cols-sql/2/train-2-num.sql --test-query-file ../train-test-data/imdb-cols-sql/2/test-2-num.sql --min-max-file ./data/col2_min_max_vals.csv --train')
    os.system(
        'python train.py --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../train-test-data/imdb-cols-sql/2/train-2-num.sql --test-query-file ../train-test-data/imdb-cols-sql/2/test-2-num.sql --min-max-file ./data/col2_min_max_vals.csv')
elif cols == 4:
    os.system(
        'python train.py --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../train-test-data/imdb-cols-sql/4/train-4-num.sql --test-query-file ../train-test-data/imdb-cols-sql/4/test-only4-num.sql --min-max-file ./data/col4_min_max_vals.csv --train')
    os.system(
        'python train.py --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../train-test-data/imdb-cols-sql/4/train-4-num.sql --test-query-file ../train-test-data/imdb-cols-sql/4/test-only4-num.sql --min-max-file ./data/col4_min_max_vals.csv')
elif cols == 6:
    os.system(
        'python train.py --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../train-test-data/imdb-cols-sql/6/train-6-num.sql --test-query-file ../train-test-data/imdb-cols-sql/6/test-only6-num.sql --min-max-file ./data/col6_min_max_vals.csv --train')
    os.system(
        'python train.py --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../train-test-data/imdb-cols-sql/6/train-6-num.sql --test-query-file ../train-test-data/imdb-cols-sql/6/test-only6-num.sql --min-max-file ./data/col6_min_max_vals.csv')
elif cols == 8:
    os.system(
        'python train.py --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../train-test-data/imdb-cols-sql/8/train-8-num.sql --test-query-file ../train-test-data/imdb-cols-sql/8/test-only8-num.sql --min-max-file ./data/col8_min_max_vals.csv --train')
    os.system(
        'python train.py --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../train-test-data/imdb-cols-sql/8/train-8-num.sql --test-query-file ../train-test-data/imdb-cols-sql/8/test-only8-num.sql --min-max-file ./data/col8_min_max_vals.csv')

elif xt == 2:
    os.system(
        'python train.py --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../train-test-data/xtzx-data-sql/2/train-2-num.sql --test-query-file ../train-test-data/xtzx-data-sql/2/test-2-num.sql --min-max-file ./data/xtcol2_min_max_vals.csv --train')
    os.system(
        'python train.py --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../train-test-data/xtzx-data-sql/2/train-2-num.sql --test-query-file ../train-test-data/xtzx-data-sql/2/test-2-num.sql --min-max-file ./data/xtcol2_min_max_vals.csv')
elif xt == 4:
    os.system(
        'python train.py --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../train-test-data/xtzx-data-sql/4/train-4-num.sql --test-query-file ../train-test-data/xtzx-data-sql/4/test-only4-num.sql --min-max-file ./data/xtcol4_min_max_vals.csv --train')
    os.system(
        'python train.py --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../train-test-data/xtzx-data-sql/4/train-4-num.sql --test-query-file ../train-test-data/xtzx-data-sql/4/test-only4-num.sql --min-max-file ./data/xtcol4_min_max_vals.csv')
elif xt == 6:
    os.system(
        'python train.py --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../train-test-data/xtzx-data-sql/6/train-6-num.sql --test-query-file ../train-test-data/xtzx-data-sql/6/test-only6-num.sql --min-max-file ./data/xtcol6_min_max_vals.csv --train')
    os.system(
        'python train.py --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../train-test-data/xtzx-data-sql/6/train-6-num.sql --test-query-file ../train-test-data/xtzx-data-sql/6/test-only6-num.sql --min-max-file ./data/xtcol6_min_max_vals.csv')
elif xt == 8:
    os.system(
        'python train.py --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../train-test-data/xtzx-data-sql/8/train-8-num.sql --test-query-file ../train-test-data/xtzx-data-sql/8/test-only8-num.sql --min-max-file ./data/xtcol8_min_max_vals.csv --train')
    os.system(
        'python train.py --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../train-test-data/xtzx-data-sql/8/train-8-num.sql --test-query-file ../train-test-data/xtzx-data-sql/8/test-only8-num.sql --min-max-file ./data/xtcol8_min_max_vals.csv')
elif xt == 10:
    os.system(
        'python train.py --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../train-test-data/xtzx-data-sql/10/train-10-num.sql --test-query-file ../train-test-data/xtzx-data-sql/10/test-only10-num.sql --min-max-file ./data/xtcol10_min_max_vals.csv --train')
    os.system(
        'python train.py --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../train-test-data/xtzx-data-sql/10/train-10-num.sql --test-query-file ../train-test-data/xtzx-data-sql/10/test-only10-num.sql --min-max-file ./data/xtcol10_min_max_vals.csv')

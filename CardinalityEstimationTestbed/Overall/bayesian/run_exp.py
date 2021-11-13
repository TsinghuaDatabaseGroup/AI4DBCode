import argparse
import os

parser = argparse.ArgumentParser(description='generatedata')
parser.add_argument('--cols', type=int, help='datasets_dir', default=4)
args = parser.parse_args()
cols = args.cols

if cols == 2:
    os.system(
        'python eval_model.py --test-file-path ../train-test-data/imdb-cols-sql/2/test-2-num.sql --single-data-path ../train-test-data/imdbdata-num --join-data-path ../train-test-data/join_samples --join-num-path ../quicksel/pattern2totalnum.pkl --join-sample-size 1000000 --run-bn')
if cols == 4:
    os.system(
        'python eval_model.py --test-file-path ../train-test-data/imdb-cols-sql/2/test-only4-num.sql --single-data-path ../train-test-data/imdbdata-num --join-data-path ../train-test-data/join_samples --join-num-path ../quicksel/pattern2totalnum.pkl --join-sample-size 1000000 --run-bn')
if cols == 6:
    os.system(
        'python eval_model.py --test-file-path ../train-test-data/imdb-cols-sql/2/test-only6-num.sql --single-data-path ../train-test-data/imdbdata-num --join-data-path ../train-test-data/join_samples --join-num-path ../quicksel/pattern2totalnum.pkl --join-sample-size 1000000 --run-bn')
if cols == 8:
    os.system(
        'python eval_model.py --test-file-path ../train-test-data/imdb-cols-sql/2/test-only8-num.sql --single-data-path ../train-test-data/imdbdata-num --join-data-path ../train-test-data/join_samples --join-num-path ../quicksel/pattern2totalnum.pkl --join-sample-size 1000000 --run-bn')

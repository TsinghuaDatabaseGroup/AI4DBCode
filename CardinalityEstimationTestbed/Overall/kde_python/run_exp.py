import argparse
import os

parser = argparse.ArgumentParser(description='generatedata')
parser.add_argument('--cols', type=int, help='cols', default=4)
args = parser.parse_args()
cols = args.cols

if cols == 2:
    os.system('python kde.py --train-file ../train-test-data/imdb-cols-sql/2/train-2-num.sql --test-file ../train-test-data/imdb-cols-sql/2/test-2-num.sql --min-max-file ../learnedcardinalities-master/data/column_min_max_vals.csv --version cols2')

elif cols == 4:
    os.system('python kde.py --train-file ../train-test-data/imdb-cols-sql/4/train-4-num.sql --test-file ../train-test-data/imdb-cols-sql/4/test-only4-num.sql --min-max-file ../learnedcardinalities-master/data/column_min_max_vals.csv --version cols4')

elif cols == 6:
    os.system('python kde.py --train-file ../train-test-data/imdb-cols-sql/6/train-6-num.sql --test-file ../train-test-data/imdb-cols-sql/6/test-only6-num.sql --min-max-file ../learnedcardinalities-master/data/column_min_max_vals.csv --version cols6')

elif cols == 8:
    os.system('python kde.py --train-file ../train-test-data/imdb-cols-sql/8/train-8-num.sql --test-file ../train-test-data/imdb-cols-sql/8/test-only8-num.sql --min-max-file ../learnedcardinalities-master/data/column_min_max_vals.csv --version cols8')
    
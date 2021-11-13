cd mscn
python preprocessing-job.py --datasets-dir ../../train-test-data/imdbdata-num/ --raw-query-file ../../train-test-data/imdb-cols-sql/4/train-4-num.sql --min-max-file ../data/col4_min_max_vals.csv
python preprocessing-job.py --datasets-dir ../../train-test-data/imdbdata-num/ --raw-query-file ../../train-test-data/imdb-cols-sql/4/test-only4-num.sql --min-max-file ../data/col4_min_max_vals.csv
cd ..
python train.py --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../train-test-data/imdb-cols-sql/4/train-4-num.sql --test-query-file ../train-test-data/imdb-cols-sql/4/test-only4-num.sql --min-max-file ./data/col4_min_max_vals.csv --train
python train.py --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../train-test-data/imdb-cols-sql/4/train-4-num.sql --test-query-file ../train-test-data/imdb-cols-sql/4/test-only4-num.sql --min-max-file ./data/col4_min_max_vals.csv

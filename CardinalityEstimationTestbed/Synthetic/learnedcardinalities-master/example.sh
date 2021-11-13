cd mscn
'python preprocessing.py --datasets-dir ./csvdata_sql --raw-query-file ../sql_truecard/' + version + \
    'train.sql' + ' --min-max-file ../CardinalityEstimationBenchmark/learnedcardinalities-master/data/' + version + '_min_max_vals.csv ' + '--table ' + version + ' --alias cdcs'
'python preprocessing.py --datasets-dir ./csvdata_sql --raw-query-file ../sql_truecard/' + version + \
    'test.sql' + ' --min-max-file ../CardinalityEstimationBenchmark/learnedcardinalities-master/data/' + version + '_min_max_vals.csv ' + '--table ' + version + ' --alias cdcs'
cd ..
python train.py --min-max-file /home/jintao/CardinalityEstimationBenchmark/learnedcardinalities-master/data/skew4_min_max_vals.csv --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file /home/jintao/CardinalityEstimationBenchmark/train-test-data/skew-sql/skew4/train-num.sql --test-query-file /home/jintao/CardinalityEstimationBenchmark/train-test-data/skew-sql/skew4/test-num.sql --train
python train.py --min-max-file /home/jintao/CardinalityEstimationBenchmark/learnedcardinalities-master/data/skew4_min_max_vals.csv --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file /home/jintao/CardinalityEstimationBenchmark/train-test-data/skew-sql/skew4/train-num.sql --test-query-file /home/jintao/CardinalityEstimationBenchmark/train-test-data/skew-sql/skew4/test-num.sql

'python preprocessing.py --datasets-dir ./csvdata_sql --raw-query-file ../sql_truecard/' + version + \
    'train.sql' + ' --min-max-file ../CardinalityEstimationBenchmark/learnedcardinalities-master/data/' + version + '_min_max_vals.csv ' + '--table ' + version + ' --alias cdcs'
'python preprocessing.py --datasets-dir ./csvdata_sql --raw-query-file ../sql_truecard/' + version + \
    'test.sql' + ' --min-max-file ../CardinalityEstimationBenchmark/learnedcardinalities-master/data/' + version + '_min_max_vals.csv ' + '--table ' + version + ' --alias cdcs'
'python train.py --min-max-file ../CardinalityEstimationBenchmark/learnedcardinalities-master/data/' + version + \
    '_min_max_vals.csv --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../sql_truecard/' + version + \
    'train.sql --test-query-file ../sql_truecard/' + version + 'test.sql --train'
'python train.py --min-max-file ../CardinalityEstimationBenchmark/learnedcardinalities-master/data/' + version + \
    '_min_max_vals.csv --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../sql_truecard/' + version + \
    'train.sql --test-query-file ../sql_truecard/' + version + 'test.sql'
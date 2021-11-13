# xgbimdb
python update.py --train-file /home/zhangjintao/CardBenchmark-Revision/Update/updated-data/imdbtrain.sql --test-file /home/zhangjintao/CardBenchmark-Revision/Update/updated-data/imdbtest.sql --min-max-file /home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master/data/imdb_min_max_vals.csv --model xgb --version imdb
# nnimdb
python update.py --train-file /home/zhangjintao/CardBenchmark-Revision/Update/updated-data/imdbtrain.sql --test-file /home/zhangjintao/CardBenchmark-Revision/Update/updated-data/imdbtest.sql --min-max-file /home/zhangjintao/Benchmark3/CardinalityEstimationBenchmark/learnedcardinalities-master/data/imdb_min_max_vals.csv --model nn --version imdb

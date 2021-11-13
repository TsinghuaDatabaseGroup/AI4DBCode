cd mscn
python preprocessing.py --datasets-dir ../../train-test-data/forest_power-data-sql/ --raw-query-file ../../train-test-data/forest_power-data-sql/foresttrain.sql --min-max-file ../data/forest_min_max_vals.csv --table forest --alias forest
python preprocessing.py --datasets-dir ../../train-test-data/forest_power-data-sql/ --raw-query-file ../../train-test-data/forest_power-data-sql/foresttest.sql --min-max-file ../data/forest_min_max_vals.csv  --table forest --alias forest
cd ..
python train.py --min-max-file ./data/forest_min_max_vals.csv --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../train-test-data/forest_power-data-sql/foresttrain.sql --test-query-file ../train-test-data/forest_power-data-sql/foresttest.sql --train --version forest
python train.py --min-max-file ./data/forest_min_max_vals.csv --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../train-test-data/forest_power-data-sql/foresttrain.sql --test-query-file ../train-test-data/forest_power-data-sql/foresttest.sql --version forest

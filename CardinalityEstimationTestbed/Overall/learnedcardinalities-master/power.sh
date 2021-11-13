cd mscn
python preprocessing.py --datasets-dir ../../train-test-data/forest_power-data-sql/ --raw-query-file ../../train-test-data/forest_power-data-sql/powertrain.sql --min-max-file ../data/power_min_max_vals.csv --table power --alias power
python preprocessing.py --datasets-dir ../../train-test-data/forest_power-data-sql/ --raw-query-file ../../train-test-data/forest_power-data-sql/powertest.sql --min-max-file ../data/power_min_max_vals.csv  --table power --alias power
cd ..
python train.py --min-max-file ./data/power_min_max_vals.csv --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../train-test-data/forest_power-data-sql/powertrain.sql --test-query-file ../train-test-data/forest_power-data-sql/powertest.sql --train --version power
python train.py --min-max-file ./data/power_min_max_vals.csv --queries 10000 --epochs 100 --batch 1024 --hid 256 --train-query-file ../train-test-data/forest_power-data-sql/powertrain.sql --test-query-file ../train-test-data/forest_power-data-sql/powertest.sql --version power

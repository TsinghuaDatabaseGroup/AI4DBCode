# For experiments that require multiple table joins, you need to generate join_samples first with factorized_sampler.py
# cp -r ../train-test-data/imdbdata-num/*.csv ./factorized_sampler/neurocard/datasets/job/
# cd ./factorized_sampler/neurocard
# python factorized_sampler.py
python eval_model.py --test-file-path ../train-test-data/imdb-cols-sql/2/test-2-num.sql --single-data-path ../train-test-data/imdbdata-num --join-data-path ../train-test-data/join_samples --join-num-path ../quicksel/pattern2totalnum.pkl --join-sample-size 1000000 --run-bn

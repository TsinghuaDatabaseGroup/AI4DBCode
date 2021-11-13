#!/bin/bash
for version in cols_4_distinct_1000_corr_6_skew_6
    do
        python3 eval_model.py --dataset=${version} --num-queries=60 --run-bn
done

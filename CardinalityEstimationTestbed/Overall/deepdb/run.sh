# Change the schema which in deepdb-imdb/deepdb_job_ranges/schemas/imdb/schema.py/def gen_job_ranges_imdb_schema first
# Modify the parameters below: query_file_location
cd deepdb-imdb/deepdb_job_ranges
python3 maqp.py --csv_path ../../../train-test-data/imdbdata-num/no_head --generate_hdf --generate_sampled_hdfs --generate_ensemble
python3 maqp.py --csv_path ../../../train-test-data/imdbdata-num/no_head --evaluate_cardinalities --rdc_spn_selection --max_variants 1 --target_path ./baselines/cardinality_estimation/results/deepdblight/imdb_light_model_based_budget_5.csv --ensemble_location ../imdb-benchmark/spn_ensembles/ensemble_join_3_budget_5_10000000.pkl --query_file_location ../../../train-test-data/imdb-cols-sql/4/test-only4-num.sql --ground_truth_file_location ./benchmarks/job-light/sql/true_cardinalities.csv

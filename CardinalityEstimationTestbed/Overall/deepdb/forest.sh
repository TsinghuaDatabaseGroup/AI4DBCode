# Change the schema which in deepdb-imdb/deepdb_job_ranges/schemas/imdb/schema.py/def gen_job_ranges_imdb_schema first
# Modify parameters for your own: ensemble_path, target_path, ensemble_location
cd deepdb-forest_power/deepdb_job_ranges
python3 maqp.py --generate_hdf --generate_sampled_hdfs --generate_ensemble --ensemble_path ../../../train-test-data/forest_power-data-sql/ --version forest
python3 maqp.py --evaluate_cardinalities --rdc_spn_selection --max_variants 1 --pairwise_rdc_path ../imdb-benchmark/spn_ensembles/pairwise_rdc.pkl --dataset imdb-ranges --target_path ../../../train-test-data/forest_power-data-sql/foresttest.sql.deepdb.results.csv --ensemble_location ../../../train-test-data/forest_power-data-sql/forest.sql.deepdb.model.pkl --query_file_location ../../../train-test-data/forest_power-sql/foresttest.sql --ground_truth_file_location ./benchmarks/job-light/sql/true_cardinalities.csv --version forest

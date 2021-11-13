import argparse
import os

parser = argparse.ArgumentParser(description='generatedata')
parser.add_argument('--cols', type=int, help='cols', default=4)
parser.add_argument('--samples', type=int, help='samples', default=4)
parser.add_argument('--xt', type=int, help='datasets_dir', default=4)
args = parser.parse_args()
cols = args.cols
samples = args.samples
xt = args.xt

if cols == 2:
    os.chdir('./deepdb-imdb/deepdb_job_ranges')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/imdbdata-num/no_head --generate_hdf --generate_sampled_hdfs --generate_ensemble --dataset cols2')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/imdbdata-num/no_head --evaluate_cardinalities --rdc_spn_selection --max_variants 1 --target_path ./baselines/cardinality_estimation/results/deepdblight/imdb_light_model_based_budget_5.csv --ensemble_location ../imdb-benchmark/spn_ensembles/ensemble_join_3_budget_5_10000000.pkl --query_file_location ../../../train-test-data/imdb-cols-sql/2/test-2-num.sql --ground_truth_file_location ./benchmarks/job-light/sql/true_cardinalities.csv --dataset cols2')
    os.chdir('..')

elif cols == 4:
    os.chdir('./deepdb-imdb/deepdb_job_ranges')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/imdbdata-num/no_head --generate_hdf --generate_sampled_hdfs --generate_ensemble --dataset cols4')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/imdbdata-num/no_head --evaluate_cardinalities --rdc_spn_selection --max_variants 1 --target_path ./baselines/cardinality_estimation/results/deepdblight/imdb_light_model_based_budget_5.csv --ensemble_location ../imdb-benchmark/spn_ensembles/ensemble_join_3_budget_5_10000000.pkl --query_file_location ../../../train-test-data/imdb-cols-sql/4/test-only4-num.sql --ground_truth_file_location ./benchmarks/job-light/sql/true_cardinalities.csv --dataset cols4')
    os.chdir('..')

elif cols == 6:
    os.chdir('./deepdb-imdb/deepdb_job_ranges')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/imdbdata-num/no_head --generate_hdf --generate_sampled_hdfs --generate_ensemble --dataset cols6')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/imdbdata-num/no_head --evaluate_cardinalities --rdc_spn_selection --max_variants 1 --target_path ./baselines/cardinality_estimation/results/deepdblight/imdb_light_model_based_budget_5.csv --ensemble_location ../imdb-benchmark/spn_ensembles/ensemble_join_3_budget_5_10000000.pkl --query_file_location ../../../train-test-data/imdb-cols-sql/6/test-only6-num.sql --ground_truth_file_location ./benchmarks/job-light/sql/true_cardinalities.csv --dataset cols6')
    os.chdir('..')

elif cols == 8:
    os.chdir('./deepdb-imdb/deepdb_job_ranges')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/imdbdata-num/no_head --generate_hdf --generate_sampled_hdfs --generate_ensemble --dataset cols8')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/imdbdata-num/no_head --evaluate_cardinalities --rdc_spn_selection --max_variants 1 --target_path ./baselines/cardinality_estimation/results/deepdblight/imdb_light_model_based_budget_5.csv --ensemble_location ../imdb-benchmark/spn_ensembles/ensemble_join_3_budget_5_10000000.pkl --query_file_location ../../../train-test-data/imdb-cols-sql/8/test-only8-num.sql --ground_truth_file_location ./benchmarks/job-light/sql/true_cardinalities.csv --dataset cols8')
    os.chdir('..')

elif samples == 3:
    os.chdir('./deepdb-imdb/deepdb_job_ranges')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/imdbdata-num/no_head --generate_hdf --generate_sampled_hdfs --generate_ensemble --samples_per_spn 10000 10000 1000 1000 1000')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/imdbdata-num/no_head --evaluate_cardinalities --rdc_spn_selection --max_variants 1 --target_path ./baselines/cardinality_estimation/results/deepdblight/imdb_light_model_based_budget_5.csv --ensemble_location ../imdb-benchmark/spn_ensembles/ensemble_join_3_budget_5_10000000.pkl --query_file_location ../../../train-test-data/imdb-cols-sql/4/test-only4-num.sql --ground_truth_file_location ./benchmarks/job-light/sql/true_cardinalities.csv')
    os.chdir('..')

elif samples == 4:
    os.chdir('./deepdb-imdb/deepdb_job_ranges')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/imdbdata-num/no_head --generate_hdf --generate_sampled_hdfs --generate_ensemble --samples_per_spn 100000 100000 10000 10000 10000')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/imdbdata-num/no_head --evaluate_cardinalities --rdc_spn_selection --max_variants 1 --target_path ./baselines/cardinality_estimation/results/deepdblight/imdb_light_model_based_budget_5.csv --ensemble_location ../imdb-benchmark/spn_ensembles/ensemble_join_3_budget_5_10000000.pkl --query_file_location ../../../train-test-data/imdb-cols-sql/4/test-only4-num.sql --ground_truth_file_location ./benchmarks/job-light/sql/true_cardinalities.csv')
    os.chdir('..')

elif samples == 5:
    os.chdir('./deepdb-imdb/deepdb_job_ranges')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/imdbdata-num/no_head --generate_hdf --generate_sampled_hdfs --generate_ensemble --samples_per_spn 1000000 1000000 100000 100000 100000')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/imdbdata-num/no_head --evaluate_cardinalities --rdc_spn_selection --max_variants 1 --target_path ./baselines/cardinality_estimation/results/deepdblight/imdb_light_model_based_budget_5.csv --ensemble_location ../imdb-benchmark/spn_ensembles/ensemble_join_3_budget_5_10000000.pkl --query_file_location ../../../train-test-data/imdb-cols-sql/4/test-only4-num.sql --ground_truth_file_location ./benchmarks/job-light/sql/true_cardinalities.csv')
    os.chdir('..')

elif samples == 6:
    os.chdir('./deepdb-imdb/deepdb_job_ranges')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/imdbdata-num/no_head --generate_hdf --generate_sampled_hdfs --generate_ensemble --samples_per_spn 10000000 10000000 1000000 1000000 1000000')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/imdbdata-num/no_head --evaluate_cardinalities --rdc_spn_selection --max_variants 1 --target_path ./baselines/cardinality_estimation/results/deepdblight/imdb_light_model_based_budget_5.csv --ensemble_location ../imdb-benchmark/spn_ensembles/ensemble_join_3_budget_5_10000000.pkl --query_file_location ../../../train-test-data/imdb-cols-sql/4/test-only4-num.sql --ground_truth_file_location ./benchmarks/job-light/sql/true_cardinalities.csv')
    os.chdir('..')

# XTZX
elif xt == 2:
    os.chdir('./deepdb-imdb/deepdb_job_ranges')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/xtzx-data-sql/no_head --generate_hdf --generate_sampled_hdfs --generate_ensemble --dataset xt2')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/xtzx-data-sql/no_head --evaluate_cardinalities --rdc_spn_selection --max_variants 1 --target_path ./baselines/cardinality_estimation/results/deepdblight/imdb_light_model_based_budget_5.csv --ensemble_location ../imdb-benchmark/spn_ensembles/ensemble_join_3_budget_5_10000000.pkl --query_file_location ../../../train-test-data/xtzx-data-sql/2/test-2-num.sql --ground_truth_file_location ./benchmarks/job-light/sql/true_cardinalities.csv --dataset xt2')
    os.chdir('..')

elif xt == 4:
    os.chdir('./deepdb-imdb/deepdb_job_ranges')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/xtzx-data-sql/no_head --generate_hdf --generate_sampled_hdfs --generate_ensemble --dataset xt4')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/xtzx-data-sql/no_head --evaluate_cardinalities --rdc_spn_selection --max_variants 1 --target_path ./baselines/cardinality_estimation/results/deepdblight/imdb_light_model_based_budget_5.csv --ensemble_location ../imdb-benchmark/spn_ensembles/ensemble_join_3_budget_5_10000000.pkl --query_file_location ../../../train-test-data/xtzx-data-sql/4/test-only4-num.sql --ground_truth_file_location ./benchmarks/job-light/sql/true_cardinalities.csv --dataset xt4')
    os.chdir('..')

elif xt == 6:
    os.chdir('./deepdb-imdb/deepdb_job_ranges')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/xtzx-data-sql/no_head --generate_hdf --generate_sampled_hdfs --generate_ensemble --dataset xt6')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/xtzx-data-sql/no_head --evaluate_cardinalities --rdc_spn_selection --max_variants 1 --target_path ./baselines/cardinality_estimation/results/deepdblight/imdb_light_model_based_budget_5.csv --ensemble_location ../imdb-benchmark/spn_ensembles/ensemble_join_3_budget_5_10000000.pkl --query_file_location ../../../train-test-data/xtzx-data-sql/6/test-only6-num.sql --ground_truth_file_location ./benchmarks/job-light/sql/true_cardinalities.csv --dataset xt6')
    os.chdir('..')

elif xt == 8:
    os.chdir('./deepdb-imdb/deepdb_job_ranges')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/xtzx-data-sql/no_head --generate_hdf --generate_sampled_hdfs --generate_ensemble --dataset xt8')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/xtzx-data-sql/no_head --evaluate_cardinalities --rdc_spn_selection --max_variants 1 --target_path ./baselines/cardinality_estimation/results/deepdblight/imdb_light_model_based_budget_5.csv --ensemble_location ../imdb-benchmark/spn_ensembles/ensemble_join_3_budget_5_10000000.pkl --query_file_location ../../../train-test-data/xtzx-data-sql/8/test-only8-num.sql --ground_truth_file_location ./benchmarks/job-light/sql/true_cardinalities.csv --dataset xt8')
    os.chdir('..')

elif xt == 10:
    os.chdir('./deepdb-imdb/deepdb_job_ranges')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/xtzx-data-sql/no_head --generate_hdf --generate_sampled_hdfs --generate_ensemble --dataset xt10')
    os.system(
        'python3 maqp.py --csv_path ../../../train-test-data/xtzx-data-sql/no_head --evaluate_cardinalities --rdc_spn_selection --max_variants 1 --target_path ./baselines/cardinality_estimation/results/deepdblight/imdb_light_model_based_budget_5.csv --ensemble_location ../imdb-benchmark/spn_ensembles/ensemble_join_3_budget_5_10000000.pkl --query_file_location ../../../train-test-data/xtzx-data-sql/10/test-only10-num.sql --ground_truth_file_location ./benchmarks/job-light/sql/true_cardinalities.csv --dataset xt10')
    os.chdir('..')

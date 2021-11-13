import argparse
import os

parser = argparse.ArgumentParser(description='mscn')

parser.add_argument('--version', type=str, help='datasets_dir', default='cols_4_distinct_1000_corr_5_skew_5')
args = parser.parse_args()
version = args.version

# schema OK
# true_cardinalities.csv
path = "./sql_truecard/"
sql_path = path + version + "test.sql"
sql_path2 = './deepdb/deepdb_job_ranges/benchmarks/job-light/sql/' + 'true_cardinalities.csv'  # true_cardinalities 
f2 = open(sql_path2, 'w')
f2.write('query_no,query,cardinality_true\n')
i = 0
with open(sql_path, 'r') as f:
    for line in f.readlines():
        strt = line[len(line) - 10: len(line)]
        tmpindex = strt.index(',')
        strt = strt[tmpindex + 1: len(strt)]
        tmpz = str(i) + ',' + str(i + 1) + ',' + strt
        f2.write(tmpz)
        i += 1
f2.close()

pre = 'python3 maqp.py --generate_hdf --generate_sampled_hdfs --generate_ensemble --ensemble_path ../../sql_truecard/ --version ' + version
run = 'python3 maqp.py --evaluate_cardinalities --rdc_spn_selection --max_variants 1 --pairwise_rdc_path ../imdb-benchmark/spn_ensembles/pairwise_rdc.pkl --dataset imdb-ranges ' + \
      '--target_path ../../sql_truecard/' + version + 'test.sql.deepdb.results.csv ' + '--ensemble_location ../../sql_truecard/' + version + \
      '.sql.deepdb.model.pkl ' + '--query_file_location ../../sql_truecard/' + version + 'test.sql ' + '--ground_truth_file_location ./benchmarks/job-light/sql/true_cardinalities.csv --version ' + version

os.chdir('./deepdb/deepdb_job_ranges')
os.system(pre)
os.system(run)

'''
python3 maqp.py --evaluate_cardinalities --rdc_spn_selection --max_variants 1 --pairwise_rdc_path ../imdb-benchmark/spn_ensembles/pairwise_rdc.pkl --dataset imdb-ranges
 --target_path ./sql_truecard/imdb_light_model_based_budget_5.csv --ensemble_location ./sql_truecard/ensemble_join_3_budget_5_10000000.pkl
 --query_file_location ./sql_truecard/test-only2-num.sql --ground_truth_file_location ./benchmarks/job-light/sql/true_cardinalities.csv
'''

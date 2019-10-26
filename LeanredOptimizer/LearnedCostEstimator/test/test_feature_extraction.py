import unittest
from src.feature_extraction.extract_features import *
from src.training.train_and_test import *

class TestFeatureExtraction(unittest.TestCase):
    def test(self):
        dataset = load_dataset('/home/sunji/cost_estimation/test_files_open_source/imdb_data_csv')
        column2pos, indexes_id, tables_id, columns_id, physic_ops_id, compare_ops_id, bool_ops_id, table_names = prepare_dataset(dataset)
        sample_num = 1000
        sample = prepare_samples(dataset, sample_num, table_names)

        feature_extractor('/home/sunji/cost_estimation/test_files_open_source/plans.json', '/home/sunji/cost_estimation/test_files_open_source/plans_seq.json')
        add_sample_bitmap('/home/sunji/cost_estimation/test_files_open_source/plans_seq.json', '/home/sunji/cost_estimation/test_files_open_source/plans_seq_sample.json', dataset, sample, sample_num)

if __name__ == '__main__':
    unittest.main()
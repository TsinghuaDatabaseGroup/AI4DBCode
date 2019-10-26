# Learning-based-cost-estimator

Source code of Feature Encoding and Model Training for cardinality and cost estimation based on PostgreSQL Execution Plans.
The tree-structured model can generate representations for sub-plans which could be used by other query processing tasks like Join Reordering,
Materialized View and even Index Selection.

### Unitest
```bash
export PYTHONPATH=code/
python -m unittest code.test.test_feature_extraction.TestFeatureExtraction
python -m unittest code.test.test_feature_encoding.TestFeatureEncoding
python -m unittest code.test.test_training.TestTraining
```

### Test Data
We offer the test data including sampled Execution Plans from PostgreSQL,
IMDB datasets, Statistics of Minimum and Maximum number of each columns.
However, the pre-trained dictionary for string keywords is too large,
users can generate it by using the code in token_embedding
module which is hard-coded for JOB workload. 

[click here for files](https://cloud.tsinghua.edu.cn/f/930a0ab8546b407a826b/?dl=1)  
Password: end2endsun

### Citation
```
@article{sun2019endtoend,
  title={An End-to-End Learning-based Cost Estimator},
  author={Ji Sun and Guoliang Li},
  journal={arXiv preprint arXiv:1906.02560},
  year={2019}
}
```

### Contact
If you have any issue, feel free to post on [Project](https://github.com/greatji/Learning-based-cost-estimator).

# 按workload的性能预测

## 1. 数据生成
使用spark-bench-legacy生成数据。

## 2. 日志解析
解析脚本在scripts/read_history.py，生成的文件放在ml_baselines/file如：
```json
{
    "AppId":"application_1610277360178_0019",
    "AppName":"Spark SVDPlusPlus Application",
    "Duration":68533,
    "SparkParameters":[
        " spark.executor.memory=1g ",
        " spark.executor.cores=1 ",
        " spark.memory.fraction=0.2 ",
        " spark.executor.instances=1 "
    ],
    "WorkloadConf":[
        "numV=20000","NUM_OF_PARTITIONS=10","mu=4.0",
        "sigma=1.3","NUM_ITERATION=1","RANK=50",
        "MINVAL=0.0","MAXVAL=5.0","GAMMA1=0.007",
        "GAMMA2=0.007","GAMMA6=0.005","GAMMA7=0.015",
        "SPARK_STORAGE_MEMORYFRACTION=0.5"
    ]
}
```
## 3. 数据合并
第二步的数据为一个workload一个文件，需要合并为一个完整的数据集。

合并方法在 data_process.py/merger_data，将合并后的文件保存为csv。

## 4. 添加其他特征
前三步生成的为基本特征和输出，不包含spark程序的信息，可依需要添加。

添加代码TF-IDF过程：
1. ml_baselines/code_process.py对spark-bench中的代码处理为TF-IDF特征，保存为npy文件，在ml_baselines/tf-idf文件夹。
2. ml_baselines/data_process.py/read_merged_data_with_code_tfidf方法，将TF-IDF特征concat到输入。

## 5. 训练与评估预测模型
ml_baselines/main.py，使用了sk-learn的多种回归模型来预测。


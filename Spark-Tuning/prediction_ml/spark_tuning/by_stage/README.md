# 按stage数据集处理过程

## 1. 处理原始日志
scripts/history_by_stage.py

用于解析spark生成的原始日志文件。其中与workload level不同的在于，获取了每个stage的运行时间、读写数据量。
生成文件存于build_dataset/dataset_by_workload下。每行为一个完整workload的执行信息，如下：
```json
{
    "AppId":"application_1603868249556_1765",
    "AppName":"DecisionTree classification Example",
    "Duration":124742,
    "SparkParameters":[
        " spark.executor.memory=1g ",
        " spark.executor.cores=1 ",
        " spark.memory.fraction=0.1 ",
        " spark.executor.instances=1 "
    ],
    "StageInfo":{
        "0":{"duration":1091,"input":65536,"output":0,"read":0,"write":0},
        "1":{"duration":14628,"input":601332046,"output":0,"read":0,"write":0}
    },
    "WorkloadConf":[
        "NUM_OF_EXAMPLES=1000000","NUM_OF_FEATURES=30","NUM_OF_CLASS_C=20",
        "impurityC=\"gini\"","maxDepthC=10","maxBinsC=100","modeC=\"Classification\"",
        "NUM_OF_CLASS_R=10","impurityR=\"variance\"","maxDepthR=5","maxBinsR=100",
        "modeR=\"Regression\"","MAX_ITERATION=3","SPARK_STORAGE_MEMORYFRACTION=0.79"
    ]
}
```
## 2. 数据切分与合并
build_dataset/build_dataset.py

第一步生成的数据仍是按workload的，需要将每一行都切分为多个stage。然后将所有workload的所有stage的数据合并为一个完整数据集, 存储为csv格式，在build_dataset/dataset_by_stage_merged文件夹。可以按需要将merged_df分割训练集和测试集。

注：此处为了方便，将每个stage中spark日志detail中的指示的那行代码也放入csv中的一列"code"。获取的代码的方法在get_code.py，从raw-log即spark的原始日志中获取。


# 按stage性能预测过程
此处对workload代码特征进行了多种尝试：使用spark日志指示的stage的代码、使用Java instrumentation对spark程序动态插桩获取每个stage所调用的所有代码、对stage调用的所有代码进行过滤等。

## 训练
### 1. 使用spark日志指示的单行代码
data_process_one_line.py

由于数据集处理过程已经将此代码放入csv文件，此处直接对code列处理即可。可以生成TF-IDF特征，或者n-gram特征。
处理完毕存储为npy文件，在 ml_baselines/npy_dataset_one_line/。

运行ml_baselines/main.py，选择npy_dataset_one_line数据集，即可。

### 2. 使用插桩获取的每个stage调用的所有代码
instrumentation/all_code_by_stage/get_all_code.py 根据插桩日志，从spark代码库中搜索代码并按stage保存于instrumentation/all_code。

ml_baselines/data_process_all.py

### 3. 插桩代码过滤
ml_baselines/data_process_all_filtered.py 将instrumentation/log中的插桩代码，使用mutual information进行过滤，过滤后文件存储于instrumentation/log_filtered。

instrumentation/all_code_by_stage/get_all_code.py 与上面几乎一致，替换路径即可。

ml_baselines/data_process_all.py 与使用全部代码几乎完全一致，只需要替换掉all_code_dir，替换npy数据的路径为npy_dataset_all_filtered/即可。

### 4. 使用DAG结点代码


## 预测
ml_baselines/predict_workload.py 注意数据处理需要与训练数据完全一致。

<div align= "center">
    <h1>Grep: A Graph Learning Based Database Partitioning System</h1>
</div>

<p align="center">
  <a href="#1-setup">Setup</a> •
  <a href="#2-unit-tests">Unit Tests</a> •
  <a href="#3-model-training">Model Training</a> •
  <a href="#4-frontend-demo">Frontend Demo</a> •  
  <a href="#citation">Citation</a> •
</p>

Grep is a database partitioning framework using graph embedding algorithms, which judiciously selects partition-keys for each table in order to maximize the performance. Grep includes four parts, i.e., *partition-key selection*, *selected-key evaluation*, *model training*, and *frontend demo*.

- **A demo of Grep**

<video width="640" height="360" controls>
  <source src="grep_6_20.mp4" type="video/mp4">
</video>


## 1. Setup

### DB Cluster

Implementing a distributed database (cluster) is actually a tricky stuff. And we recommend you to implement GreenPlum, which is open-source and much more stable than some classic distributed databases like PG-XL. You can refer to our [install_db_cluster.md](install_db_cluster.md) for implementation instructions.

### Packages

```bash
pip install -r requirements.txt
```

## 2. Unit Tests

### Partition-Key Selection

Step 1: Change the settings within ./api/services/partition/config.py.

Step 2: Run the test script that selects partition keys withhout evaluation feedback.

```bash
python test_partition_key_selection.py
```

### Selected-Key Evaluation

Step 1: Change the settings within ./api/services/partition/config.py.

Step 2: Run the test script that estimate the performance under selected partitioning keys.

```bash
python test_partition_key_evaluation.py
```

## 3. Model Training

### Self-supervised Training 

```bash
python train_partition_models.py
```

### Supervised Training

TBD

## 4. Frontend Demo

### Backend

```bash
python app.py
```

### Frontend

```bash
cd   web/
npm install
npm run dev
```



## Citation

If you use Grep in your research, please cite:

```bibtex
@article{zhou2023grep,
  title={Grep: A Graph Learning Based Database Partitioning System},
  author={Zhou, Xuanhe and Li, Guoliang and Feng, Jianhua and Liu, Luyang and Guo, Wei},
  journal={Proceedings of the ACM on Management of Data},
  volume={1},
  number={1},
  pages={1--24},
  year={2023},
  publisher={ACM New York, NY, USA}
}
```
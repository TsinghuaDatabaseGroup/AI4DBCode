<div align= "center">
    <h1>Grep: A Graph Learning Based Database Partitioning System</h1>
</div>

<p align="center">
  <a href="#1-setup">Setup</a> •
  <a href="#2-unit-tests">Unit Tests</a> •
  <a href="#3-model-training">Model Training</a> •
  <a href="#4-web-application">Web Application</a> •  
  <a href="#5-practices">Practices</a> •  
  <a href="#citation">Citation</a> •
</p>

Grep is a database partitioning framework using graph embedding algorithms, which judiciously selects partition-keys for each table in order to maximize the performance. Grep includes four parts, i.e., *partition-key selection*, *selected-key evaluation*, *model training*, and *frontend demo*.

*A complete version will be made available at https://github.com/zhouxh19/grep*

- **A demo of Grep**

https://github.com/TsinghuaDatabaseGroup/AI4DBCode/assets/17394639/9bebbdde-d0d6-42b7-a501-63caedb7a020

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

## 4. Web Application

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

## 5. Practices

Check out the subset of queries and partiton results (only those that are publicly available) within *./practices*.


## Citation

If you use Grep in your research, please cite:

```bibtex
@article{DBLP:journals/pacmmod/ZhouLFLG23,
  author       = {Xuanhe Zhou and
                  Guoliang Li and
                  Jianhua Feng and
                  Luyang Liu and
                  Wei Guo},
  title        = {Grep: {A} Graph Learning Based Database Partitioning System},
  journal      = {Proc. {ACM} Manag. Data},
  volume       = {1},
  number       = {1},
  pages        = {94:1--94:24},
  year         = {2023},
  url          = {https://doi.org/10.1145/3588948},
  doi          = {10.1145/3588948},
  timestamp    = {Mon, 19 Jun 2023 16:36:09 +0200},
  biburl       = {https://dblp.org/rec/journals/pacmmod/ZhouLFLG23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

<div align="center">

-----
Grep: A Graph Learning Based Database Partitioning System.

[**What is Grep?**](#-what-is-grep)
</div>

## What is Grep?

We propose a partitioning key selection framework using graph embedding algorithms. This framework mainly includes three parts:

1. **Build column graph:**
We characterize behaviors of queries on the columns in the form of a graph model, where the vertices denote the features of columns and the edges capture query costs among different columns (e.g., equi-joins). We generate column graphs with different workloads and store them as training data.

2. **Select partitioning keys:**
We adopt a graph-based learning model to embed graph features for every column and select columns based on the embedded subgraphs.

    **Training.** We first choose a combination of graph embedding (e.g., simple GCN) and relevance decomposition as the key selection model. And then, with column graphs as training samples, for each graph, we iteratively use the graph model to select partitioning keys and utilize the performance to tune model parameters.

    **Inference.** Given a column graph (V, E), we input (V, E) into the trained model and the model gives a column vector, where 1 represents the corresponding column is selected and 0 is not.

3. **Evaluate partitioning performance:**
To reduce the partitioning overhead, we pre-train a graph-representaion-based evaluation model to estimate the workload performance for each partitioning strategy.

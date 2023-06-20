import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import psycopg2
from api.services.partition.models.gnn import GNN
import math
import re

splitted_queries = []


class ComputeNode:
    def __init__(self, joins, points):  # points == filters + aggregates
        self.joins = joins
        self.points = points


def remove_alias(predicate):
    table_prefixes = re.findall(r'\b\w+\.', predicate)

    tbl_id = 0
    for prefix in table_prefixes:
        if '\'' in prefix or '"' in prefix:
            continue
        predicate = predicate.replace(prefix, 'alias{}.'.format(tbl_id))
        tbl_id = tbl_id + 1

    return predicate


def parse_query_plan(query_plan):
    global splitted_queries

    node_type = query_plan['Node Type']
    sql = ""

    if node_type == 'Hash Join':
        # Get the join condition
        join_cond = query_plan['Hash Cond']
        join_cond = join_cond.replace('(', '').replace(')', '').replace('::numeric', '')
        join_cond = remove_alias(join_cond)

        sql = f"SELECT * FROM ({parse_query_plan(query_plan['Plans'][0])}) AS alias0 JOIN ({parse_query_plan(query_plan['Plans'][1])}) AS alias1 ON {join_cond}"

    elif node_type == 'Nested Loop':
        # Get the join condition
        join_cond = query_plan['Join Filter']

        sql = f"SELECT * FROM ({parse_query_plan(query_plan['Plans'][0])}) JOIN ({parse_query_plan(query_plan['Plans'][1])}) ON {join_cond.replace('(', '').replace(')', '').replace('::numeric', '')}"

    elif node_type == 'Seq Scan':
        # Get the relation name and alias
        rel_name = query_plan['Relation Name']
        alias = query_plan['Alias']

        # Generate SQL statement
        if 'Filter' not in query_plan:
            sql = f"SELECT * FROM {rel_name} AS {alias}"
        else:
            sql = f"SELECT * FROM {rel_name} AS {alias} WHERE {query_plan['Filter'].replace('::numeric', '')}"

    elif node_type == 'Hash':
        # Generate SQL statements for each sub-plan
        sql = parse_query_plan(query_plan['Plans'][0])

    else:
        if "Plans" in query_plan:
            for sub_plan in query_plan['Plans']:
                parse_query_plan(sub_plan)

    if sql != '' and sql + ';' not in splitted_queries:
        splitted_queries.append(sql + ';')

    return sql


def is_sql(sql):
    sql = sql.lower()

    if 'select' in sql and 'from' in sql:
        return True
    return False


class SampleGraph:
    def __init__(self, args, partitioning_keys=None, is_sample=False):
        global splitted_queries

        self.partitioning_keys = partitioning_keys
        self.args = args

        self.source_db_credentials = {
            "host": self.args.db_host,
            "port": self.args.db_port,
            "dbname": self.args.database,
            "user": self.args.db_user,
            "password": self.args.db_password,
        }

        self.destination_db_credentials = {
            "host": self.args.sample_db_host,
            "port": self.args.sample_db_port,
            "dbname": self.args.sample_db_name,
            "user": self.args.sample_db_user,
            "password": self.args.sample_db_password,
        }

        # [ok] Sample data into a new database

        # [ok] split the origin query into multiple single-operator queries 

        # [ok] modify the single-operator queries: for each query, if it does not include partitioned tables, remove that query; otherwise, add the key columns in the partitioned tables to the project clause

        # [ok]: based on the results and corresponding operators, initialize the overall vertex matrix and edge matrix

        # [ok] execute these queries and obtain the results to compute the vertex vector and edge weights

        # [todo] ensure the atomic query is executable

        # [todo] check whether cloned database exists

        self.atomic_queries = self.split_queries_into_single_operators()
        # ["SELECT * FROM lineitem AS l WHERE (l_quantity > '10');", 'SELECT * FROM orders AS o;', 'SELECT * FROM customer AS c;', 'SELECT * FROM (SELECT * FROM orders AS o) AS alias0 JOIN (SELECT * FROM customer AS c) AS alias1 ON alias0.o_custkey = alias1.c_custkey;', "SELECT * FROM (SELECT * FROM lineitem AS l WHERE (l_quantity > '10')) AS alias0 JOIN (SELECT * FROM (SELECT * FROM orders AS o) AS alias0 JOIN (SELECT * FROM customer AS c) AS alias1 ON alias0.o_custkey = alias1.c_custkey) AS alias1 ON alias0.l_orderkey = alias1.o_orderkey AND alias0.l_shipdate = alias1.o_orderdate;"]

        # self.atomic_queries = self.add_key_columns(atomic_queries)
        # self.atomic_queries = ['SELECT customer.c_custkey FROM customer AS customer;', 'SELECT orders.o_custkey, orders.o_orderdate FROM orders AS orders;', 'SELECT c_custkey, o_custkey, o_orderdate FROM (SELECT * FROM customer) AS customer JOIN (SELECT * FROM orders) AS orders ON c_custkey = o_custkey;', "SELECT lineitem.l_orderkey, lineitem.l_quantity FROM lineitem AS lineitem WHERE (l_quantity > '10');", "SELECT o_custkey, o_orderdate, l_orderkey, l_quantity FROM (SELECT * FROM orders) AS orders JOIN (SELECT * FROM lineitem WHERE (l_quantity > '10')) AS lineitem ON o_orderkey = l_orderkey AND orders.o_orderdate = lineitem.l_shipdate;"]

        print(" ==== splitted queries: ", self.atomic_queries)

        # execute the atomic queries and build the graph based on the query results
        self.vertex_matrix, self.edge_matrix, self.vertex_json = self.build_sample_graph(is_sample)

        '''
        if self.partitioning_keys is not None:

            self.sampled_table_tuples = self.load_data()
            """
            {'lineitem':    l_orderkey
                    0           1
                    1           1
                    2           1, 
             'orders':      o_orderkey    o_custkey    o_orderdate
                    0           1             1         1996-01-02 }
            """
            
            self.sampled_table_tuples = self.allocate_tuples(self.sampled_table_tuples, self.args.node_num)
            """
            {'lineitem':    l_quantity  node_id
            0        36.0      0.0
            1        32.0      0.0
            2        28.0      0.0, 'orders':    o_custkey  node_id
            0          1      1.0}            
            """
        
            # Initialize an empty dictionary for subgraphs
            subgraphs = {}

            # Iterate through the tables and their corresponding DataFrames in sampled_table_tuples
            for table, df in self.sampled_table_tuples.items():
                # Group the DataFrame rows by the 'node_id' column
                grouped = df.groupby('node_id')

                # Iterate through the groups (node_id, sub_df) and add them to the subgraphs
                for node_id, sub_df in grouped:
                    # If the subgraph with the current node_id does not exist, initialize it with an empty dictionary
                    if node_id not in subgraphs:
                        subgraphs[node_id] = {}

                    # Add the table DataFrame to the subgraph with the current node_id
                    subgraphs[node_id][table] = sub_df.drop('node_id', axis=1)

            # Print the subgraphs
            for node_id, subgraph in subgraphs.items():
                print(f"Subgraph {node_id}:")
                for table, sub_df in subgraph.items():
                    print(f"  Table {table}:")
                    print(sub_df)

            # Initialize the subgraphs for each node
            self.subgraphs = [{} for _ in range(self.args.node_num)]
            for i in range(self.args.node_num):
                if i in subgraphs:
                    self.subgraphs[i]['vertex_matrix'] = self.init_vertex_matrix(subgraphs[i])
                    self.subgraphs[i]['edge_matrix'] = self.init_edge_matrix(self.subgraphs[i]['vertex_matrix'].shape[0])
            
            # Initialize the inter-subgraph edge matrix across all nodes
            self.inter_subgraph_edge_matrix = self.init_edge_matrix(self.args.node_num)

        else:
            # singular database
            pass        
        '''

    def init_adjacant_matrix(self, size):
        return np.array([[0] * size] * size)

    def build_sample_graph(self, is_sample):

        # 1. Initialize the graph
        columns = [ComputeNode(0, 0) for i in range(self.args.node_num)]
        vertex_matrix = np.array([[col.joins, col.points] for col in columns])

        edge_matrix = self.init_adjacant_matrix(self.args.node_num)

        # 3. Execute the atomic queries and build the graph based on the query results
        destination_conn = psycopg2.connect(**self.destination_db_credentials)
        destination_cursor = destination_conn.cursor()

        # [join]
        # [filter, project, aggregate, sort, groupby, limit, distinct, union, intersect, except]
        for query in self.atomic_queries:
            destination_cursor.execute(query)
            rows = destination_cursor.fetchall()
            colnames = [desc[0] for desc in destination_cursor.description]

            query_partitioning_key_positions = self.match_partitioning_keys(colnames)

            if len(query_partitioning_key_positions) == 1:
                for i, row in enumerate(rows):
                    node_num = self.hash_partitioning(row, query_partitioning_key_positions[0], self.args.node_num)
                    vertex_matrix[node_num][1] += 1
            elif len(query_partitioning_key_positions) > 1:
                for row in rows:
                    node_num_0 = self.hash_partitioning(row, query_partitioning_key_positions[0], self.args.node_num)
                    node_num_1 = self.hash_partitioning(row, query_partitioning_key_positions[1], self.args.node_num)
                    if node_num_0 != node_num_1:
                        edge_matrix[node_num_0][node_num_1] += 1
                        edge_matrix[node_num_1][node_num_0] += 1
                    else:
                        vertex_matrix[node_num_0][0] += 1
            elif query_partitioning_key_positions == []:  # (assume) task is evenly distributed
                for i, row in enumerate(rows):
                    vertex_matrix[i % self.args.node_num][1] += 1

        vertex_list = vertex_matrix.tolist()
        vertex_json = {}
        for i, col in enumerate(vertex_list):
            vertex_json["node_"+str(i)] = {"joins": col[0], "points": col[1]}

        vertex_matrix = torch.tensor(vertex_matrix, dtype=torch.float32)
        edge_matrix = torch.tensor(edge_matrix, dtype=torch.float32)

        return vertex_matrix, edge_matrix, vertex_json

    def hash_partitioning(self, tuple, partitioning_key_list, k):
        node_idx = 0
        for key in partitioning_key_list:
            node_idx += hash(tuple[key]) % k  # todo replace with other partitioning function

        return node_idx % k

    def match_partitioning_keys(self, colnames):
        # {'lineitem': ['l_orderkey', 'l_quantity'], 'orders': ['o_custkey', 'o_orderdate'], 'customer': ['c_custkey']}
        matched_partitioning_keys = []

        for table in self.partitioning_keys:
            keys = self.partitioning_keys[table]
            keys = keys.split(',')
            contains_all_keys = True
            key_pos = []
            for key_column in keys:
                if key_column not in colnames:
                    contains_all_keys = False
                    break
                else:
                    # get the position of key_column in colnames
                    key_column_idx = colnames.index(key_column)
                    key_pos.append(key_column_idx)

            if contains_all_keys:
                matched_partitioning_keys.append(key_pos)

        return matched_partitioning_keys

    def split_queries_into_single_operators(self):
        global splitted_queries

        self.workload = []

        with open(self.args.workload_path, 'r') as f:
            file_contents = f.read()

            queries = file_contents.split(';')
            for query in queries:
                if is_sql(query):
                    self.workload.append(query)

        conn = psycopg2.connect(**self.destination_db_credentials)
        cur = conn.cursor()

        for query in self.workload:
            # Execute the SQL query and obtain the query plan
            cur.execute("""EXPLAIN (FORMAT JSON) {}""".format(query))

            # Get the query plan
            query_plan = cur.fetchall()

            parse_query_plan(query_plan[0][0][0]["Plan"])

        return splitted_queries

    def contains_partitioned_table(self, query, partitioned_tables):
        for table in partitioned_tables:
            if table in query:
                return True
        return False

    def add_columns(self, query, partitioning_keys):
        for table, key_columns in partitioning_keys.items():
            if table in query:
                columns_str = ', '.join(f"{table}.{col}" for col in key_columns)
                query = re.sub(r'(SELECT\s+)', rf'\1{columns_str}, ', query, flags=re.IGNORECASE)
        return query

    def add_key_columns(self, atomic_queries):

        new_atomic_queries = []
        for query in atomic_queries:
            if self.contains_partitioned_table(query, self.partitioning_keys):
                new_atomic_queries.append(self.add_columns(query, self.partitioning_keys))

        return new_atomic_queries

    def init_vertex_matrix(self, subgraph):
        # vertex feature: row_position, filter frequency
        print("=====================================")
        print(subgraph)

        vertex_num = 0
        for table in subgraph:
            vertex_num += subgraph[table].shape[0]

        vertex_matrix = np.zeros((vertex_num, 2))  # filters, aggregates

        # Extract 'row_position' and copy it to the first column of vertex_matrix
        row_index = 0
        for table in subgraph:
            row_positions = subgraph[table]['row_position'].values
            vertex_matrix[row_index:row_index + len(row_positions), 0] = row_positions
            row_index += len(row_positions)

        print("Updated vertex_matrix:")
        print(vertex_matrix)

        return vertex_matrix

    def init_edge_matrix(self, vertex_num):

        return np.array([[0] * vertex_num] * vertex_num)

    def graph_construct(self):
        # Sample 0.01% tuples from the origin database

        # Allocate sampled tuples into the k nodes
        nodes = self.allocate_tuples(self.sampled_table_tuples, self.args.node_num)

        # Compute edge weights (join frequencies) and vertex weights
        edge_weights = self.compute_edge_weights(nodes)

        return nodes, edge_weights

    def compute_edge_weights(self, nodes):  # nodes: {node_idx: [tuple1, tuple2, ...]}
        # Compute edge weights (join frequencies)
        edge_weights = {}
        for node_idx, tuples in nodes.items():
            for i in range(len(tuples)):
                for j in range(i + 1, len(tuples)):
                    edge = (tuple(tuples[i]), tuple(tuples[j]))  # Convert Series to tuple
                    if edge not in edge_weights:
                        edge_weights[edge] = 1
                    else:
                        edge_weights[edge] += 1

        return edge_weights

    def load_data(self):
        # Connect to the origin database

        conn = psycopg2.connect(
            host=self.args.db_host,
            database=self.args.database,
            user=self.args.db_user,
            password=self.args.db_password,
            port=self.args.db_port)

        sampled_table_tuples = {}
        # self.partitioning_keys = {'customer': ['c_custkey']}
        row_num = 0
        for table in self.partitioning_keys:
            # Sample self.args.sample_ratio tuples from table
            # stratified sampling
            # columns = ','.join(self.partitioning_keys[table])
            sql_query = """
                WITH total_rows AS (
                SELECT COUNT(*) as count FROM {}
                ),
                sample_size AS (
                SELECT CEIL({} * count) AS size FROM total_rows
                )
                SELECT * 
                FROM {}
                ORDER BY random()
                LIMIT (SELECT size FROM sample_size);        
            """.format(table, self.args.sample_ratio, table)

            data = pd.read_sql_query(sql_query, conn)
            data['row_position'] = [i + row_num for i, row in enumerate(data.iterrows())]

            row_num += len(data)

            sampled_table_tuples[table] = data

        conn.close()

        print(sampled_table_tuples)

        return sampled_table_tuples

    def allocate_tuples(self, sample_data, k):

        for table in sample_data:
            sampled_tuples = sample_data[table]
            partitioning_key_list = self.partitioning_keys[table]
            for index, row in sampled_tuples.iterrows():
                # compute the node_idx based on the consistent hash of all the keys in partitioning_key_list
                node_idx = self.hash_partitioning(row, partitioning_key_list, k)

                sample_data[table].at[index, 'node_id'] = node_idx

                # todo weighted frequencies of local operations (i.e., filters, aggregates, writes)

        print(sample_data)
        return sample_data

    def representation_single_node(self, subgraph):
        # Step 1 - Representation of Single Node
        # Implement aggregation of features and embedding into k vectors H(P_i)

        # Placeholder for the feature extraction and aggregation
        features = torch.Tensor()  # Replace with the actual feature extraction and aggregation
        return features

    def representation_multiple_nodes(self, vertex_matrix, edge_matrix):
        # Step 2 - Representation of Multiple Nodes
        # Generate k-vertex graph with compound vertex matrix and edge matrix
        tilde_v = torch.cat(vertex_matrix, dim=0)
        tilde_e = edge_matrix

        # Convolution on tilde_v and tilde_e
        h_p = torch.matmul(tilde_e, tilde_v)

        return h_p

    def query_performance_mapping(self, graph_vector):
        # Step 3 - Query Performance Mapping
        # Implement a fully-connected (FC) layer with ReLU activation function
        fc_layer = nn.Sequential(
            nn.Linear(graph_vector.size(1), 1),
            nn.ReLU()
        )

        performance = fc_layer(graph_vector)
        return performance

    def max_pooling(self, h_p):
        # Max pooling on h_p to pool up the partitioning vectors
        # into a graph vector
        graph_vector = torch.max(h_p, dim=0)[0].unsqueeze(0)
        return graph_vector


class partition_evaluation_model:
    # compute the embedding for the partitioned_sample_graph
    # estimate the query latency based on the embedding for the partitioned_sample_graph    
    def __init__(self, args):
        self.args = args
        self.input_dim = args.max_node_num
        self.hidden_dim = self.get_hidden_dimension(args.max_node_num)
        self.output_dim = args.max_node_num

        self.gnn = GNN(input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim)
        print(" === evaluation gnn size ({}, {}, {})".format(self.input_dim, self.hidden_dim, self.output_dim))

        self.fc_layer = nn.Sequential(
            nn.Linear(self.output_dim, 1),
            # input size of node number (from the embedding tensor), output size of 1 (for the latency value)
            nn.LeakyReLU()
        )

        nn.init.xavier_uniform_(self.fc_layer[0].weight)

    def get_hidden_dimension(self, graph_vertex):
        return max(math.ceil(2 * math.sqrt(graph_vertex * graph_vertex)), 10)

    def get_gnn_dimensions(self, joined_columns):
        input_dim = joined_columns.shape[1]
        output_dim = joined_columns.shape[1]
        hidden_dim = max(math.ceil(2 * math.sqrt(input_dim * output_dim)), 10)

        return input_dim, hidden_dim, output_dim

    def embedding(self, graph):
        # The graph vector encodes features within the {\it $k$-node sample graph}, which contains similar data and query distribution as that on the whole datasets. Hence, we further use a fully-connected (FC) layer, with ReLU as the activation function, to map the graph vector on sampled tuples into the partitioning performance on the whole dataset. The FC layer aggregates the features of the graph vector with non-linear transformations and approximates performance metrics by learning the network weights based on deviation from the actual performance.

        # separately aggregate the features \revise{within the subgraph of} each node (e.g., local joins/selects/writes) and embed them into $k$ embedding vectors $H(P_i)$ by multiplying the edge weight matrix, vertex matrix of each node with the globally-shared graph network weights, which represent the data scales \revise{(e.g, the number of vertices)} and computation costs of local operations \revise{(e.g., the edges of vertices in the same subgraph)}. 

        # each vertex vector is encoded by an embedding layer
        # embeddings = []
        # for vertex_vector in graph.vertex_matrix:
        #     vertex_vector = self.gnn(vertex_vector)
        #     embeddings.append(vertex_vector)
        embeddings, V = self.gnn(graph.vertex_matrix, graph.edge_matrix)

        # all the vertex embedded vectors are aggregated into a graph vector by max pooling
        embedding = torch.max(embeddings, dim=0)[0]

        return embedding

    def estimate_latency(self, embedding):
        # estimate the latency based on the embedding for the partitioned_sample_graph

        # latency = torch.exp(embedding) / (1 + torch.exp(embedding))
        latency = torch.abs(self.fc_layer(embedding))

        # return latency.item()
        return latency

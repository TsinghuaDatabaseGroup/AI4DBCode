import torch
import numpy as np
import math

from api.services.partition.database import database
from api.services.partition.models.attention_network import AttentionNetwork
from api.services.partition.models.gnn import GNN


class Column:
    def __init__(self, table_id, table_size, tuple_selectivity, tuple_length, num_filters, num_aggregates, num_writes):
        self.table_id = table_id
        self.table_size = table_size
        self.tuple_selectivity = tuple_selectivity
        self.tuple_length = tuple_length
        self.num_filters = num_filters
        self.num_aggregates = num_aggregates
        self.num_writes = num_writes

class Column2Graph:
    def __init__(self, args):
        self.args = args
        print("Loading workload...", self.args.schema_path)
        self.schema = database.table_col(self.args.schema_path)
        self.cols = {}
        for tbl in self.schema:
            for col in self.schema[tbl]:
                if col not in self.cols:
                    self.cols[col.lower()] = tbl.lower()
        
        with open(self.args.workload_path, 'r') as f:
            file_contents = f.read()

            queries = file_contents.split(';')
            self.workload = [query.strip() for query in queries]

        # fetch and index the tables
        self.table_info = database.extract_tables(self.args)

        ## specify the involved columns
        self.used_cols = {}
        for col in self.cols:
            res = self.is_col_used(col=col, workload=self.workload)

            if res is True:
                self.used_cols[col] = self.cols[col]

        # compute the distnct value ratio for each used col
        self.col_distinct_value_ratio = {}
        for col in self.used_cols:
            self.col_distinct_value_ratio[col] = database.distinct_value_ratio(col, self.used_cols[col], self.args)


        self.filter_predicates = []
        self.join_predicates = []

        for sql in self.workload:
            filters, joins = database.extract_predicates(sql)
            self.filter_predicates   += filters
            self.join_predicates     += joins

        self.weighted_vertex_matrix, self.edge_matrix, self.vertex_json, self.edge_json = self.build_graph()

        self.vertex_matrix = self.weighted_vertex_matrix

    def init_adjacant_matrix(self, size):
        return np.array([[0]*size]*size)

    def is_col_used(self, col, workload):
        for sql in workload:
            if col in sql:
                return True
        return False
    
    def add_f(self, col, col_times):
        if col not in col_times:
            col_times[col] = 1
        else:
            col_times[col] = col_times[col] + 1

    def parse_workload(self, joins, filters, joined_column_matrix, used_cols, columns):

        print(used_cols)

        cols_list = list(used_cols.keys())

        for predicate in joins:
            print(predicate)
            left_col, right_col = predicate.split(' = ')
            
            left_table, left_col_name = left_col.split('.')
            right_table, right_col_name = right_col.split('.')
            
            left_col_name = left_col_name.lower()
            left_col_name = left_col_name.strip()
            left_col_pos = cols_list.index(left_col_name)
            right_col_name = right_col_name.lower()
            right_col_name = right_col_name.strip()
            right_col_pos = cols_list.index(right_col_name)
            
            joined_column_matrix[left_col_pos][right_col_pos] = database.join_card_on_sampled_data(predicate, used_cols[left_col_name], used_cols[right_col_name], left_table, right_table, self.args) # cardinality of the join

        for predicate in filters:
            for col in used_cols:
                if col in predicate:
                    columns[col][4] = columns[col][4] + 1 # num_filters

        return joined_column_matrix, columns

    # Implement the graph construction method based on EdgeWeight and VertexWeight descriptions
    def build_graph(self):
        # Define the graph

        #todo Example columns
        columns = {}

        for col in self.used_cols:
            # create Column for each column
            table_id = int(self.table_info[self.used_cols[col]]["table_id"])
            table_size = float(self.table_info[self.used_cols[col]]["table_size"].split(" ")[0])
            tuple_selectivity = float(self.col_distinct_value_ratio[col])
            tuple_length = int(self.table_info[self.used_cols[col]]["tuple_length"])
            num_filters = 0
            num_aggregates = 0
            num_writes = 0
            columns[col] = [table_id, table_size, tuple_selectivity, tuple_length, num_filters, num_aggregates, num_writes]

        edge_matrix = self.init_adjacant_matrix(len(self.used_cols))
        edge_matrix, columns = self.parse_workload(self.join_predicates, self.filter_predicates, edge_matrix, self.used_cols, columns)
        
        for i in range(len(self.used_cols)):
            for j in range(len(self.used_cols)):
                if edge_matrix[i][j] != 0:
                    edge_matrix[j][i] = edge_matrix[i][j]
        edge_matrix = torch.tensor(edge_matrix, dtype=torch.float32)

        used_cols_list = list(self.used_cols.keys())

        edge_list = edge_matrix.tolist()
        edge_json = {}
        edge_count = 0
        for i in range(len(edge_list)):
            for j in range(i, len(edge_list[i])):
                if edge_list[i][j] != 0:
                    edge_json[str(edge_count)] = {"source": used_cols_list[i], "target": used_cols_list[j], "value": edge_list[i][j]}
                    edge_count = edge_count + 1

        vertex_matrix = np.array([[columns[col][0], columns[col][1], columns[col][2], columns[col][3], columns[col][4], columns[col][5], columns[col][6]] for col in columns])
        vertex_matrix = torch.tensor(vertex_matrix, dtype=torch.float32)
        vertex_list = vertex_matrix.tolist()
        vertex_json = {}
        for i in range(len(vertex_list)):
            vertex_json[used_cols_list[i]] = vertex_list[i][2]

        attention_network = AttentionNetwork(input_size=7)
        attention_weights = attention_network(vertex_matrix, edge_matrix)

        weighted_vertex_matrix = vertex_matrix
        for i in range(vertex_matrix.shape[0]):
            weighted_vertex_matrix[i] = attention_weights[i] * vertex_matrix[i]

        return weighted_vertex_matrix, edge_matrix, vertex_json, edge_json

    def normalized_laplacian(self):
        # Compute the degree matrix
        degree_matrix = torch.diag(torch.sum(self.edge_matrix, dim=1))

        # Compute the inverse square root of the degree matrix
        degree_matrix_sqrt_inv = torch.diag(1.0 / torch.sqrt(torch.diag(degree_matrix)))

        # Compute the normalized Laplacian matrix
        laplacian = torch.eye(self.edge_matrix.shape[0]) - torch.matmul(torch.matmul(degree_matrix_sqrt_inv, self.edge_matrix), degree_matrix_sqrt_inv)

        return laplacian


class partitioning_model:

    def __init__(self, args):
        super(partitioning_model, self).__init__()

        # Define the layers for the Partitioning Model based on the descriptions
        #self.input_dim, self.hidden_dim, self.output_dim = self.get_gnn_dimensions(graph.edge_matrix)
        self.args = args

        hidden_dim = self.get_hidden_dimension(args.selection_graph_vertex)

        self.gnns = []
        for i in range(args.partition_gnn_layers):
            gnn = GNN(input_dim=args.selection_graph_vertex, 
                       hidden_dim=hidden_dim, 
                       output_dim=args.selection_graph_vertex)
            gnn.train(True)            
            self.gnns.append(gnn)

        self.grads = {}  # Initialize grads here

        print(" === partition gnn size ({}, {}, {})".format(args.selection_graph_vertex, hidden_dim, args.selection_graph_vertex))
        
    def padding_graph_matrcies(self, vertex_matrix, edge_matrix):

        #vertex_matrix = torch.tensor(vertex_matrix, dtype=torch.float32)

        current_size = vertex_matrix.size(0)
        target_size = self.args.selection_graph_vertex

        if current_size < target_size:
            pad_size = target_size - current_size
            padding = torch.zeros((pad_size,) + vertex_matrix.shape[1:], dtype=torch.float32)
            padded_vertex_matrix = torch.cat((vertex_matrix, padding), dim=0)

            padding = torch.zeros((pad_size, current_size), dtype=torch.float32)
            padded_edge_matrix = torch.cat([edge_matrix, padding], dim=0)
            padding = torch.zeros((target_size, pad_size), dtype=torch.float32)
            padded_edge_matrix = torch.cat([padded_edge_matrix, padding], dim=1)

        else:
            # raise an alart
            print("The number of the dataset columns is larger than the maximium threshold!")
            exit(1)

        return padded_vertex_matrix, padded_edge_matrix


    def compute_gradient(self, p_loss):
        # First we need to clear any existing gradients

        self.grads = {}  # Reset the dictionary at the start of the computation

        for gnn in self.gnns:
            for param in gnn.parameters():
                if param.grad is not None:
                    param.grad.data.zero_()

        # Add regularization term to the loss
        l2_reg = 0

        for gnn in self.gnns:
            for param in gnn.parameters():
                l2_reg += torch.norm(param)

        reg_loss = p_loss + self.args.reg_factor * l2_reg  # self.args.reg_factor is the regularization factor

        for gnn in self.gnns:
            for name, param in gnn.named_parameters():
                derivative = torch.autograd.grad(reg_loss, param, create_graph=True, allow_unused=True)
                if derivative[0] is not None:
                    self.grads[name] = derivative[0]

        with torch.no_grad():
            for gnn in self.gnns:
                for name, param in gnn.named_parameters():
                    if name in self.grads:
                        param -= self.args.partition_learning_rate * self.grads[name]


    def get_hidden_dimension(self, graph_vertex):

        return max(math.ceil(2 * math.sqrt(graph_vertex * graph_vertex)), 10)

    # def get_gnn_dimensions(self, joined_columns):
    #     input_dim = joined_columns.shape[1]
    #     output_dim = joined_columns.shape[1]
    #     hidden_dim = max(math.ceil(2 * math.sqrt(input_dim * output_dim)), 10)

    #     return input_dim, hidden_dim, output_dim

    def forward(self, graph):

        padded_vertex_matrix, padded_edge_matrix = self.padding_graph_matrcies(graph.vertex_matrix, graph.edge_matrix)
        origin_graph = graph 

        # Implement the forward pass for the Partitioning Model
        # Compute embeddings

        embeddings = padded_vertex_matrix
        for gnn in self.gnns:
            embeddings, V = gnn(embeddings, padded_edge_matrix)  # Get the original embeddings

        # embeddings = embeddings[:origin_graph.vertex_matrix.size(0), :]
        # V = V[:origin_graph.vertex_matrix.size(0), :]

        total_relevance = torch.ones(embeddings.shape[0])

        # Compute partial derivatives of relevance with respect to each input feature

        for gnn in reversed(self.gnns):
            gnn_weights= gnn.W[:embeddings.size(-1), :]
            partial_derivatives = gnn_weights.t() * total_relevance


        # Compute relevance for each input feature
        relevance = partial_derivatives[:embeddings.size(-1), :] * embeddings
        total_relevance = relevance.sum(dim=1)

        # Compute partitioning benefits for each column in the origin graph
        partitioning_benefits = total_relevance[:origin_graph.vertex_matrix.size(0)]

        # Compute probability of using each column as the partitioning key
        probabilities = torch.exp(partitioning_benefits) / (1 + torch.exp(partitioning_benefits))
        partitioning_keys = (probabilities > 0.5).int()

        return partitioning_keys


import os


class PartitionConfig:
    def __init__(self):
        self.exp_id = "adv_exp_id"
        self.node_num = 2
        self.max_node_num = 10
        self.selection_graph_vertex = 150
        self.partition_learning_rate = 0.1
        self.partition_gnn_layers = 1
        self.evaluation_learning_rate = 0.1
        self.reload_pretrain = False
        self.pretrain_model_checkpoint = "tpch_demo_1686802593"
        self.saved_model_dir = "saved_models"
        self.reg_factor = 12.5
        self.sample_ratio = 0.01
        self.data_size = 1
        self.workload_concurrency = 1
        self.server_script_path = "/home/xuanhe/datasets"
        self.server = "xxx"
        self.server_username = "xxx"
        self.server_password = "xxx"
        self.server_port = 17211
        self.database = "tpch_demo"
        self.schema = "tpch_demo"
        self.db_user = "xxx"
        self.db_password = "xxx"
        self.db_host = "xxx"
        self.db_port = "xxx"
        self.sample_db_user = "xxx"
        self.sample_db_password = "xxx"
        self.sample_db_name = "sample_data"
        self.sample_db_host = "xxx"
        self.sample_db_port = "xxx"
        self.target_db_user = "xxx"
        self.target_db_password = "xxx"
        self.target_db_host = "xxx"
        self.target_db_port = "xxx"
        self.training_epochs = 100
        self.raw_data_dir = "/home/xuanhe/tpch-kit-master/dbgen/generated-tpch-data"
        self.use_estimated_results = False  # 不需要指定        
        self.schema_path = ""  # 不需要指定
        self.workload_path = ""  # 不需要指定

    def generate_paths(self):
        current_file = os.path.abspath(__file__)
        parent_path = os.path.dirname(current_file)
        self.workload_path = os.path.join(
            parent_path, "datasets/" + self.database + "/workload.sql")
        self.schema_path = os.path.join(
            parent_path,
            "datasets/" +
            self.database +
            "/schema.sql")
        self.saved_model_dir = os.path.join(parent_path, self.saved_model_dir)

        if not os.path.exists(self.workload_path):
            return False, "workload_path {} does not exist.".format(
                self.workload_path)

        if not os.path.exists(self.schema_path):
            return False, "schema_path {} does not exist.".format(
                self.schema_path)

        if self.reload_pretrain:
            self.pretrain_model_checkpoint = os.path.join(
                parent_path, self.saved_model_dir, self.pretrain_model_checkpoint)
            if not os.path.exists(self.pretrain_model_checkpoint):
                return False, "saved_model_dir {} does not exist.".format(
                    self.pretrain_model_checkpoint)

        return True, "success"

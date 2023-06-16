import os


class PartitionConfig:
    def __init__(self):
        self.exp_id = "adv_exp_id"
        self.node_num = 2
        self.max_node_num = 10
        self.selection_graph_vertex = 150
        self.partition_learning_rate = 0.1
        self.evaluation_learning_rate = 0.1
        self.reload_pretrain = False
        self.pretrain_model_checkpoint = "tpch_demo_1686800752"
        self.saved_model_dir = "saved_models"
        self.reg_factor = 12.5
        self.sample_ratio = 0.01
        self.data_size = 1
        self.server_script_path = "/home/xuanhe/datasets"
        self.server = "66.111.121.55"
        self.server_username = "xuanhe"
        self.server_password = "80b361292d39"
        self.server_port = 17211
        self.database = "tpch_demo"
        self.schema = "tpch_demo"
        self.db_user = "gpadmin"
        self.db_password = "80b361292d39"
        self.db_host = "166.111.121.55"
        self.db_port = "54322"
        self.sample_db_user = "postgres"
        self.sample_db_password = "kDZCNgUV0zJwdq9"
        self.sample_db_name = "sample_data"
        self.sample_db_host = "166.111.121.55"
        self.sample_db_port = "15432"
        self.target_db_user = "gpadmin"
        self.target_db_password = "80b361292d39"
        self.target_db_host = "166.111.121.55"
        self.target_db_port = "54322"
        self.training_epochs = 100
        self.raw_data_dir = "/home/xuanhe/tpch-kit-master/dbgen/generated-tpch-data"
        self.use_estimated_results = True  # 不需要指定        
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

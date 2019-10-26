
class Parameters():
    def __init__(self, condition_max_num, indexes_id, tables_id, columns_id, physic_ops_id, column_total_num,
                 table_total_num, index_total_num, physic_op_total_num, condition_op_dim, compare_ops_id, bool_ops_id,
                 bool_ops_total_num, compare_ops_total_num, data, min_max_column, word_vectors, cost_label_min,
                 cost_label_max, card_label_min, card_label_max):
        self.condition_max_num = condition_max_num
        self.indexes_id = indexes_id
        self.tables_id = tables_id
        self.columns_id = columns_id
        self.physic_ops_id = physic_ops_id
        self.column_total_num = column_total_num
        self.table_total_num = table_total_num
        self.index_total_num = index_total_num
        self.physic_op_total_num = physic_op_total_num
        self.condition_op_dim = condition_op_dim
        self.compare_ops_id = compare_ops_id
        self.bool_ops_id = bool_ops_id
        self.bool_ops_total_num = bool_ops_total_num
        self.compare_ops_total_num = compare_ops_total_num
        self.data = data
        self.min_max_column = min_max_column
        self.word_vectors = word_vectors
        self.cost_label_min = cost_label_min
        self.cost_label_max = cost_label_max
        self.card_label_min = card_label_min
        self.card_label_max = card_label_max
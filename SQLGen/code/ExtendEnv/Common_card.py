import base
import numpy as np
from cost.sample_data import MetaDataSupport
from treelib import Tree

np.set_printoptions(threshold=np.inf)

# support grammar key word
operator = ['=', '!=', '>', '<', '<=', '>=']
order_by_key = ['DESC', 'ASC']
# predicate_type = ['between', 'is null', 'is not null', 'in', 'is not in', 'exists', 'not exists', 'like', 'not like']
# conjunction = ['and', 'or']
# predicate_exist = ['exist', 'not exist']
predicate_in = ['in', 'not in']
conjunction = ['and']
aggregate = ['max', 'min', 'avg', 'sum']
# keyword = ['select', 'from', 'aggregate', 'where', 'group by', 'having', 'order by']
keyword = ['query', 'update', 'delete', 'insert', 'select', 'from', 'aggregate', 'where', 'having', 'order by',
           'subquery', 'into', 'set']  # group by是被迫的去掉了


class DataNode(object):
    # action_index 与 identifier不同, action_index是map里面的，identifire是tree里面的
    def __init__(self, action_index, datatype=None, key_type=None):
        self.action_index = action_index
        self.datatype = datatype
        self.key_type = key_type


class Common(object):
    step_reward = 0
    bug_reward = -100
    start_word = "@"
    terminal_word = " "     # sql terminal word
    sub_terminal_word = "#"     # subquery terminal word
    SEQ_LENGTH = 40

    def __init__(self, metric, dbname, target_type, server_name='postgresql', allowed_error=0.1):
        """
        Initialization constant
        :param metric:
        :param dbname:
        :param target_type:
        :param server_name:
        :param allowed_error:
        """
        self.allowed_error = allowed_error
        self.target_type = target_type  # target_type(0: point, 1:range)
        self.metric = metric
        if target_type == 0:
            self.target = metric
            self.task_name = f"card_pc{metric}"
            # self.log_target = np.log1p(self.target)
        else:
            # self.low_b = np.log1p(metric[0])
            # self.up_b = np.log1p(metric[1])
            self.low_b = metric[0]
            self.up_b = metric[1]
            self.task_name = f"card_rc{self.low_b}_{self.up_b}"

        self.dbname = dbname
        self.server_name = server_name

        # self.db, self.cursor = base.connect_server(dbname, server_name=server_name)
        self.SampleData = MetaDataSupport(dbname)
        self.schema = self.SampleData.schema
        self.word_num_map, self.num_word_map, self.relation_tree = self._build_relation_env()

        self.relation_graph = base.build_relation_graph(self.dbname, self.schema)

        self.action_space = self.observation_space = len(self.word_num_map)
        self.operator = [self.word_num_map[x] for x in operator]
        # self.predicate_exist = [self.word_num_map[x] for x in predicate_exist]
        self.predicate_in = [self.word_num_map[x] for x in predicate_in]
        # self.order_by_key = [self.word_num_map[x] for x in order_by_key]
        # self.predicate_type = [self.word_num_map[x] for x in predicate_type]
        self.conjunction = [self.word_num_map[x] for x in conjunction]
        # self.aggregate = [self.word_num_map[x] for x in aggregate]
        self.keyword = [self.word_num_map[x] for x in keyword]
        # self.integer = [self.word_num_map[x] for x in integer]
        # self.join = [self.word_num_map[x] for x in join]

        self.attributes = []

        table_node = self.relation_tree.children(self.relation_tree.root)
        self.tables = [field.identifier for field in table_node]
        for node in table_node:
            self.attributes += [field.identifier for field in self.relation_tree.children(node.identifier)]

    @staticmethod
    def add_map(series, word_num_map, num_word_map):
        count = len(word_num_map)
        for word in series:
            if word not in word_num_map.keys():
                word_num_map[word] = count
                num_word_map[count] = word
                count += 1

    def _build_relation_env(self):
        print("_build_env")

        sample_data = self.SampleData.get_data()

        tree = Tree()
        tree.create_node("root", 0, None, data=DataNode(0))

        word_num_map = dict()
        num_word_map = dict()

        word_num_map[self.start_word] = 0
        num_word_map[0] = self.start_word

        word_num_map[self.terminal_word] = 1
        num_word_map[1] = self.terminal_word

        word_num_map[self.sub_terminal_word] = 2
        num_word_map[2] = self.sub_terminal_word

        # 第一层 table_names
        count = 3
        for table_name in self.schema.keys():
            tree.create_node(table_name, count, parent=0, data=DataNode(count, datatype="table_name"))
            word_num_map[table_name] = count
            num_word_map[count] = table_name
            count += 1

        # 第二层 table的attributes
        for table_name in self.schema.keys():
            for field in self.schema[table_name]:
                attribute = '{0}.{1}'.format(table_name, field)
                tree.create_node(attribute, count, parent=word_num_map[table_name],
                                 data=DataNode(count))
                word_num_map[attribute] = count
                num_word_map[count] = attribute
                count += 1

        # 第三层 每个taoble的sample data
        for table_name in self.schema.keys():
            for field in self.schema[table_name]:
                for data in sample_data[table_name][field]:
                    if data in word_num_map.keys():
                        pass
                    else:
                        word_num_map[data] = len(num_word_map)
                        num_word_map[len(num_word_map)] = data
                    field_name = '{0}.{1}'.format(table_name, field)
                    tree.create_node(data, count, parent=word_num_map[field_name], data=DataNode(word_num_map[data]))
                    count += 1

        self.add_map(operator, word_num_map, num_word_map)
        self.add_map(order_by_key, word_num_map, num_word_map)
        # self.add_map(predicate_type, word_num_map, num_word_map)
        self.add_map(conjunction, word_num_map, num_word_map)
        self.add_map(aggregate, word_num_map, num_word_map)
        self.add_map(keyword, word_num_map, num_word_map)
        # self.add_map(predicate_exist, word_num_map, num_word_map)
        self.add_map(predicate_in, word_num_map, num_word_map)
        # self.add_map(integer, word_num_map,num_word_map)
        # self.add_map(join, word_num_map, num_word_map)

        print("_build_env done...")
        print("action/observation space:", len(num_word_map), len(word_num_map))
        print("relation tree size:", tree.size())
        # tree.show()
        return word_num_map, num_word_map, tree

    def activate_space(self, cur_space, keyword):   # 用keyword开启 cur_space 到 next_space 的门
        # 激活下一个space
        cur_space[self.word_num_map[keyword]] = 1

    def activate_terminal(self, cur_space):
        cur_space[self.word_num_map[self.terminal_word]] = 1

    def cal_e_card(self, sql):
        # print(sql)
        result, query_info = base.get_evaluate_query_info(self.dbname, sql)
        if result != 1:
            # print(sql)
            return -1
        return query_info['e_cardinality']

    def cal_reward(self, sql):
        if self.target_type == 0:
            return self.cal_point_reward(sql)
        else:
            return self.cal_range_reward(sql)

    def is_satisfy(self, sql):
        e_card = self.cal_e_card(sql)
        assert e_card != -1
        if self.target_type == 0:
            if self.metric * (1 - self.allowed_error) <= e_card <= self.metric * (1 + self.allowed_error):
                return True
            else:
                return False
        else:
            if self.low_b <= e_card <= self.up_b:
                return True
            else:
                return False

    def cal_point_reward(self, sql):
        e_card = self.cal_e_card(sql)
        if e_card == -1:
            return self.step_reward
        else:
            reward = (-base.relative_error(e_card, self.target) + self.allowed_error) * 10
            # if reward > 0:
            #     print(self.get_sql())
            #     print("e_card:{} reward:{}".format(e_card, reward))
            reward = max(reward, -10)
            return reward

    def cal_range_reward(self, sql):
        e_card = self.cal_e_card(sql)
        if e_card == -1:
            return self.step_reward
        else:
            # print(self.get_sql())
            if self.low_b <= e_card <= self.up_b:
                # print(self.get_sql())
                # print("e_card:{} reward:{}".format(e_card, 2))
                return 5
            else:
                relative_error = max(base.relative_error(e_card, self.up_b),
                                     base.relative_error(e_card, self.low_b))
                reward = -relative_error
                reward = max(reward, -5)
                # a = min(e_card / self.up_b, self.up_b / e_card)
                # b = min(e_card / self.low_b, self.low_b / e_card)
                # reward = max(a, b)
                return reward

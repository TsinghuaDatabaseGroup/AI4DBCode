import numpy as np
from treelib import Tree
from enum import IntEnum
import math
import os
import base
import sys
from cost.sample_data import MetaDataSupport

np.set_printoptions(threshold=np.inf)

# support grammar key word

operator = ['=', '!=', '>', '<', '<=', '>=']
order_by_key = ['DESC', 'ASC']
# predicate_type = ['between', 'is null', 'is not null', 'in', 'is not in', 'exists', 'not exists', 'like', 'not like']
# predicate_type = ['between', 'like', 'not like']
# conjunction = ['and', 'or']
conjunction = ['and']
aggregate = ['max', 'min', 'avg', 'sum']
# keyword = ['select', 'from', 'aggregate', 'where', 'group by', 'having', 'order by']
keyword = ['select', 'from', 'aggregate', 'where', 'having', 'order by'] # group by是被迫的去掉了
# join = ['join']
# as_name = ['tmp1', 'tmp2', 'tmp3', 'tmp4', 'tmp5']  # as 空间得固定住
# integer = [str(i) for i in range(50)]


class DataNode(object):
    # action_index 与 identifier不同, action_index是map里面的，identifire是tree里面的
    def __init__(self, action_index, datatype=None, key_type=None):
        self.action_index = action_index
        self.datatype = datatype
        self.key_type = key_type


# 一些设定 #
# 聚合函数的出现一定会导致group by
# group by 一定程度出现having 对输出结果的控制

def cal_expect_cost():
    with open('./queries/sql_cost') as f:
        total = 0
        count = 0
        for line in f.readlines():
            line = line.strip(('\n'))
            total += float(line)
            count += 1
    f.close()
    return total / count


class GenSqlEnv(object):
    def __init__(self, metric, dbname, target_type, server_name='postgresql', allowed_error=0.1):

        # self.expect_cost = cal_expect_cost()
        # print("expect sql cost:", self.expect_cost)

        self.allowed_error = allowed_error
        self.target_type = target_type  # target_type(0: point, 1:range)
        self.metric = metric
        if target_type == 0:
            self.target = metric
            self.task_name = f"cost_pc{metric}"
            # self.log_target = np.log1p(self.target)
        else:
            # self.low_b = np.log1p(metric[0])
            # self.up_b = np.log1p(metric[1])
            self.low_b = metric[0]
            self.up_b = metric[1]
            self.task_name = f"cost_rc{self.low_b}_{self.up_b}"

        self.dbname = dbname
        self.server_name = server_name

        self.db, self.cursor = base.connect_server(dbname, server_name=server_name)

        self.step_reward = 0
        self.bug_reward = -100
        self.terminal_word = " "  # 空格结束、map在0上,index为0
        self.SampleData = MetaDataSupport(dbname)
        self.schema = self.SampleData.schema
        self.word_num_map, self.num_word_map, self.relation_tree = self._build_relation_env()

        self.relation_graph = base.build_relation_graph(self.dbname, self.schema)
        # print(self.relation_graph)

        self.action_space = self.observation_space = len(self.word_num_map)

        # self.grammar_tree = self._build_grammar_env()
        self.select_space = []
        self.from_space = []
        self.where_space = []
        self.group_by_space = []
        self.having_space = []
        self.order_by_space = []
        self.aggregate_space = []

        self.group_key = False

        self.operator = [self.word_num_map[x] for x in operator]
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

        # print(self.relation_table)
        self.select_clause = self.from_clause = self.where_clause = self.group_by_clause = self.having_clause = self.order_by_clause = self.aggregate_clause = ""

        self.master_control = {
            'select': [self.select_observe, self.select_action],
            'from': [self.from_observe, self.from_action],
            'where': [self.where_observe, self.where_action],
            # 'group by': [self.group_by_observe, self.group_by_action],
            'having': [self.having_observe, self.having_action],
            'order by': [self.order_by_observe, self.order_by_action],
            'aggregate': [self.aggregate_observe, self.aggregate_action],
        }

        self.cur_state = self.master_control['from']  # 初始时为from
        self.time_step = 0

    def _build_relation_env(self):
        print("_build_env")

        sample_data = self.SampleData.get_data()

        tree = Tree()
        tree.create_node("root", 0, None, data=DataNode(0))

        word_num_map = dict()
        num_word_map = dict()

        word_num_map[self.terminal_word] = 0
        num_word_map[0] = self.terminal_word

        # 第一层 table_names
        count = 1
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
        # self.add_map(integer, word_num_map,num_word_map)
        # self.add_map(join, word_num_map, num_word_map)

        print("_build_env done...")
        print("action/observation space:", len(num_word_map), len(word_num_map))
        print("relation tree size:", tree.size())
        # tree.show()
        return word_num_map, num_word_map, tree

    def reset(self):
        # print("reset")
        self.cur_state = self.master_control['from']
        self.select_clause = self.from_clause = self.where_clause = self.group_by_clause = self.having_clause = self.order_by_clause = self.aggregate_clause = ""
        self.where_space.clear()
        self.from_space.clear()
        self.select_space.clear()
        self.aggregate_space.clear()
        self.group_by_space.clear()
        self.order_by_space.clear()
        self.having_space.clear()
        self.time_step = 0
        self.group_key = False
        return self.word_num_map['from']

    def activate_space(self, cur_space, keyword):   # 用keyword开启 cur_space 到 next_space 的门
        # 激活下一个space
        cur_space[keyword] = 1

    def activate_ternminal(self, cur_space):
        cur_space[0] = 1

    def select_observe(self, observation):
        # self.cur_sql[: -1]
        candidate_word = np.zeros((self.action_space,), dtype=int)
        if self.num_word_map[observation] == 'select':     # 第一次进
            self.need_select_table = self.from_space.copy()
            for table_index in self.need_select_table:
                candidate_word[[field.identifier for field in self.relation_tree.children(table_index)]] = 1
            return candidate_word
        else:   # attribtue
            if self.need_select_table:
                for table_index in self.need_select_table:
                    candidate_word[[field.identifier for field in self.relation_tree.children(table_index)]] = 1
                return candidate_word
            else:   # table和普通的attribute选完了可以聚合也可以where condition 也可以orderby
                self.activate_space(candidate_word, self.word_num_map['aggregate'])
                self.activate_space(candidate_word, self.word_num_map['where'])
                self.activate_space(candidate_word, self.word_num_map['order by'])
                self.activate_ternminal(candidate_word)
                return candidate_word

    def select_action(self, action):
        # print('enter select_action:', self.num_word_map[action])
        if self.num_word_map[action] == 'select':
            self.select_clause = 'select'
        elif action in self.keyword:
            self.cur_state = self.master_control[self.num_word_map[action]]
            self.cur_state[1](action)
        else:
            self.select_space.append(action)
            self.group_by_space.append(action)
            self.order_by_space.append(action)
            table_name_index = self.relation_tree.parent(action).identifier   #
            self.need_select_table.remove(table_name_index)
            if self.select_clause == 'select':
                self.select_clause = self.select_clause + ' ' + self.num_word_map[action]
            else:
                self.select_clause = self.select_clause + ', ' + self.num_word_map[action]
        return self.cal_reward(), 0

    def aggregate_observe(self, observation=None):
        candidate_word = np.zeros((self.action_space,), dtype=int)
        if self.group_key is False:
            self.group_by_generate()    # 直接group by产生
            self.group_key = True
        # self.activate_space(candidate_word, self.word_num_map['aggregate'])
        self.activate_space(candidate_word, self.word_num_map['where'])
        self.activate_space(candidate_word, self.word_num_map['order by'])
        self.activate_space(candidate_word, self.word_num_map['having'])
        self.activate_ternminal(candidate_word)
        return candidate_word

    def aggregate_action(self, action):
        if action == self.word_num_map['aggregate']:
            table = np.random.choice(self.from_space)
            attributes = [node.identifier for node in self.relation_tree.children(table)]
            choose_attribute = np.random.choice(attributes)
            choose_aggregate_type = np.random.choice(['max', 'min', 'avg', 'sum'])
            self.aggregate_space.append((choose_aggregate_type, choose_attribute))
            self.aggregate_clause = self.aggregate_clause + ' ' + '{aggregate_type}({aggregate_attribute})'.format(
                aggregate_type=choose_aggregate_type, aggregate_attribute=self.num_word_map[choose_attribute])
        else:   # 其他key_word
            self.cur_state = self.master_control[self.num_word_map[action]]
            self.cur_state[1](action)
        return self.cal_reward(), 0

    def from_observe(self, observation=None):
        if observation == self.word_num_map['from']:    # 第一次进来
            self.from_clause = 'from'
            candidate_tables = np.zeros((self.action_space,), dtype=int)
            candidate_tables[self.tables] = 1
            return candidate_tables
        else:  # observation in self.tables:   # 选择table 激活join type
            relation_tables = self.relation_graph.get_relation(self.num_word_map[observation])  # string类型
            relation_tables = set([self.word_num_map[table] for table in relation_tables])
            relation_tables = list(relation_tables.difference(self.from_space))     # 选过的不选了
            candidate_tables = np.zeros((self.action_space,), dtype=int)
            candidate_tables[relation_tables] = 1
            if len(self.from_space) > 0:
                candidate_tables[self.word_num_map['select']] = 1
            return candidate_tables

    def from_action(self, action):
        # print("enter from action")
        if action in self.tables:
            self.from_space.append(action)
            if self.from_clause == 'from':
                self.from_clause = self.from_clause + ' ' + self.num_word_map[self.from_space[0]]
            else:
                table1 = self.from_space[len(self.from_space)-1]
                table2 = self.from_space[len(self.from_space)-2]
                relation_key = self.relation_graph.get_relation_key(self.num_word_map[table1],
                                                                    self.num_word_map[table2])
                frelation = relation_key[0]
                trelation = relation_key[1]
                join_condition = frelation[0] + '=' + trelation[0]
                for i in range(1, len(frelation)):
                    join_condition = join_condition + ' and ' + frelation[i] + '=' + trelation[i]
                self.from_clause = self.from_clause + ' join ' + self.num_word_map[table1] + ' on ' + join_condition
        elif action == self.word_num_map['select']:
            self.cur_state = self.master_control['select']
            self.cur_state[1](action)
        else:
            print('from error')
        return self.cal_reward(), 0

    def where_observe(self, observation):
        # print("enter where space")
        candidate_word = np.zeros((self.action_space,), dtype=int)
        if observation == self.word_num_map['where']:
            self.where_attributes = []
            for table_index in self.from_space:
                for field in self.relation_tree.children(table_index):
                    self.where_attributes.append(field.identifier)
            candidate_word[self.where_attributes] = 1
            return candidate_word
        elif observation in self.attributes:
            candidate_word[self.operator] = 1
            # candidate_condition[self.predicate_type] = 1
            return candidate_word
        elif observation in self.operator:
            candidate_word[self.operation_data(self.cur_attribtue)] = 1
            return candidate_word
        elif observation in self.conjunction:
            candidate_word[self.where_attributes] = 1
            return candidate_word
        else:   # data
            candidate_word[self.conjunction] = 1
            self.activate_ternminal(candidate_word)
            self.activate_space(candidate_word, self.word_num_map['order by'])
            if self.group_key:
                self.activate_space(candidate_word, self.word_num_map['having'])
            return candidate_word

        # elif observation in self.predicate_type:

    def where_action(self, action):
        # print("enter where action")
        # print(self.num_word_map[action])
        if action == self.word_num_map['where']:
            self.where_clause = 'where '
        elif action in self.attributes:
            self.cur_attribtue = action
            self.where_clause = self.where_clause + self.num_word_map[action]
        elif action in self.operator:
            self.where_clause = self.where_clause + ' ' + self.num_word_map[action] + ' '
        elif action in self.conjunction:
            self.where_clause = self.where_clause + ' {} '.format(self.num_word_map[action])
        elif action in self.keyword:
            self.cur_state = self.master_control[self.num_word_map[action]]
            self.cur_state[1](action)
        else:   # data
            self.where_clause = self.where_clause + str(self.num_word_map[action])
        return self.cal_reward(), 0

    def operation_data(self, attributes):
        data = [node.data.action_index for node in self.relation_tree.children(attributes)]
        return data

    def group_by_generate(self):
        self.group_by_clause = 'group by'
        for attribute in self.group_by_space:
            self.group_by_clause = self.group_by_clause + ' ' + self.num_word_map[attribute] + ','
        self.group_by_clause = self.group_by_clause[: -1]

    def having_observe(self, observation):
        # self.having_space是聚合函数 + terminal
        candidate_word = np.zeros((self.action_space,), dtype=int)
        cur_word = self.num_word_map[observation]
        if cur_word == 'having':
            self.activate_ternminal(candidate_word)
            self.activate_space(candidate_word, self.word_num_map['order by'])
        else:
            print("error")
        return candidate_word

    def having_action(self, action):
        # print("having action:", action, "===", self.num_word_map[action])
        if action == self.word_num_map['having']:
            attr = self.num_word_map[self.aggregate_space[0][1]]
            agg_type = self.aggregate_space[0][0]
            chosen_op = np.random.choice(operator)
            chosen_data = np.random.choice(self.operation_data(self.aggregate_space[0][1]))
            self.having_clause = 'having {}({}) {} {}'.format(agg_type, attr, chosen_op, chosen_data)
        else:
            self.cur_state = self.master_control[self.num_word_map[action]]
            self.cur_state[1](action)
        return self.cal_reward(), 0

    def order_by_observe(self, observation):
        candidate_word = np.zeros((self.action_space,), dtype=int)
        candidate_word[self.select_space] = 1
        if observation != self.word_num_map['order by']:
            self.activate_ternminal(candidate_word)
        return candidate_word

    def order_by_action(self, action):
        if action == self.word_num_map['order by']:
            self.order_by_clause = 'order by'
        else:
            self.select_space.remove(action)
            choose_order = np.random.choice(order_by_key)
            if self.order_by_clause == 'order by':
                self.order_by_clause = self.order_by_clause + ' ' + self.num_word_map[action] + ' ' + choose_order
            else:
                self.order_by_clause = self.order_by_clause + ', ' + self.num_word_map[action] + ' ' + choose_order
        return self.cal_reward(), 0

    # def order_by_observe(self, observation):
    #     candidate_word = np.zeros((self.action_space,), dtype=int)
    #     if observation == self.word_num_map['order by']:
    #         self.activate_space(candidate_word, self.word_num_map['select'])    # attribute
    #         if self.group_key:  #有聚合函数
    #             self.activate_space(candidate_word, self.word_num_map['aggregate'])
    #     else:
    #         self.activate_ternminal(candidate_word)
    #     return candidate_word
    #
    # def order_by_action(self, action):
    #     if action == self.word_num_map['order by']:
    #         self.order_by_clause = 'order by'
    #     elif action == self.word_num_map['select']:
    #         number = np.random.randint(1, len(self.select_space) + 1)
    #         attributes = np.random.choice(self.select_space, size=number, replace=False)
    #         for attribute in attributes:
    #             choose_order = np.random.choice(order_by_key)
    #             self.order_by_clause = self.order_by_clause + ' ' + self.num_word_map[attribute] + ' ' + choose_order + ','
    #     else:   # 'aggregate'
    #         number = np.random.randint(1, len(self.aggregate_space) + 1)
    #         tuple_indexes = np.random.choice(range(0, len(self.aggregate_space)), size=number, replace=False)
    #         for index in tuple_indexes:
    #             aggregate_tuple = self.aggregate_space[index]
    #             choose_order = np.random.choice(order_by_key)
    #             self.order_by_clause = self.order_by_clause + ' ' + '{}({})'.format(aggregate_tuple[0], self.num_word_map[aggregate_tuple[1]]) + ' ' + choose_order + ','
    #     return self.step_reward, 0

    def add_map(self, series, word_num_map, num_word_map):
        count = len(word_num_map)
        for word in series:
            if word not in word_num_map.keys():
                word_num_map[word] = count
                num_word_map[count] = word
                count += 1

    def observe(self, observation):
        """
        :param observation: index 就可以
        :return: 返回vocabulary_size的矩阵，单步reward
        """
        return self.cur_state[0](observation)

    def step(self, action):
        self.time_step += 1
        if action == 0:  # choose 结束：
            # return self.final_reward(), 1
            final_reward = self.cal_reward()
            # assert final_reward != 0
            return final_reward, 1
        elif action == -1:
            return self.bug_reward, 1
        else:
            return self.cur_state[1](action)

    def get_sql(self):
        # print("from clause:", self.from_clause)
        # print('select clause:', self.select_clause)
        # print('aggregate clause:', self.aggregate_clause)
        # print("where_clause:", self.where_clause)
        # print("having clause:", self.having_clause)
        # print("group by clause:", self.group_by_clause)
        # print("order_by_clause clause:", self.order_by_clause)
        final_sql = self.select_clause
        if self.aggregate_clause:
            final_sql = final_sql + ', ' + self.aggregate_clause
        final_sql = final_sql + ' ' + self.from_clause
        if self.where_clause:
            final_sql = final_sql + ' ' + self.where_clause
        if self.group_by_clause:
            final_sql = final_sql + ' ' + self.group_by_clause
        if self.having_clause:
            final_sql = final_sql + ' ' + self.having_clause
        if self.order_by_clause:
            final_sql = final_sql + ' ' + self.order_by_clause
        final_sql = final_sql + ';'
        return final_sql

    def get_cost(self):
        sql = self.get_sql()
        result, query_info = base.get_evaluate_query_info(self.dbname, sql)
        if result != 1:
            # print(sql)
            return -1
        return query_info['total_cost']

    def is_satisfy(self):
        e_cost = self.get_cost()
        if e_cost == -1:
            print(self.get_sql())
        assert e_cost != -1
        if self.target_type == 0:
            if self.metric * (1 - self.allowed_error) <= e_cost <= self.metric * (1 + self.allowed_error):
                return True
            else:
                return False
        else:
            if self.low_b <= e_cost <= self.up_b:
                return True
            else:
                return False

    def cal_reward(self):
        if self.target_type == 0:
            return self.cal_point_reward()
        else:
            return self.cal_range_reward()

    def cal_point_reward(self):
        e_cost = self.get_cost()
        # reward = -base.relative_error(log_e_cost, self.log_target) + self.allowed_error
        if e_cost == -1:
            return self.step_reward
        else:
            reward = (-base.relative_error(e_cost, self.target) + self.allowed_error) * 200
            #if reward >= self.target_type:
                # print(self.get_sql())
                # print("e_cost:{} reward:{}".format(e_cost, reward))
            reward = max(reward, -2)
            return reward

    def cal_range_reward(self):
        e_cost = self.get_cost()
        if e_cost == -1:
            return self.step_reward
        else:
            if self.low_b <= e_cost <= self.up_b:
                # print("e_cost:{} reward:{}".format(e_cost, 1))
                return 2
            else:
                # relative_error = min(base.relative_error(log_e_cost, self.up_b),
                relative_error = max(base.relative_error(e_cost, self.up_b),
                                     base.relative_error(e_cost, self.low_b))
                reward = -relative_error
                # if reward >= self.target_type:
                #     print("e_cost:{} reward:{}".format(e_cost, reward))
                reward = max(reward, -2)
                return reward

    def __del__(self):
        self.cursor.close()
        self.db.close()


def choose_action(observation):
    candidate_list = np.argwhere(observation == np.max(observation)).flatten()
    # action = np.random.choice(candidate_list, p=increase_key_probability(candidate_list, key_word_list, step))
    action = np.random.choice(candidate_list)
    return action


SEQ_LENGTH = 20


def discount_reward(r, gamma, final_r, ep_steps):
    # gamma 越大约有远见
    discounted_r = np.zeros(SEQ_LENGTH)
    discounted_r[ep_steps:] = final_r
    running_add = 0     # final_r已经存了
    for t in reversed(range(0, ep_steps)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def end_of_pretrain_episode_actions(final_reward, ep_steps, buffer, memory, index):
    discounted_ep_rs = discount_reward(buffer.rewards, 1, final_reward, ep_steps)
    # print("discounted_ep_rs:", discounted_ep_rs)
    episode = np.hstack((buffer.states, discounted_ep_rs, buffer.actions))
    memory[index, :] = episode


def pre_data(dbname, mtype, metric, nums):
    # episode = np.hstack((self.buffer.states, discounted_ep_rs, self.buffer.actions))
    # index = (self.episode - 1) % BATCH_SIZE
    # self.memory[index, :] = episode
    memory = np.zeros((nums, SEQ_LENGTH * 3))
    buffer = base.Buffer()
    # sqls = set()
    if mtype == 'point':
        env = GenSqlEnv(metric=metric, dbname=dbname, target_type=0)
    elif mtype == 'range':
        env = GenSqlEnv(metric=metric, dbname=dbname, target_type=1)
    else:
        print('error')
        return
    tcount = 0
    scount = 0
    while scount < nums:
        tcount += 1
        current_state = env.reset()
        reward, done = env.bug_reward, False
        ep_steps = 0
        while not (done or ep_steps >= SEQ_LENGTH):
            candidate_action = env.observe(current_state)
            action = choose_action(candidate_action)
            reward, done = env.step(action)
            buffer.store(current_state, action, reward, ep_steps)  # 单步为0
            ep_steps += 1
            current_state = action
        if ep_steps == SEQ_LENGTH or reward == env.bug_reward or reward < 0:
            buffer.clear()  # 采样忽略
            # print('采样忽略')
        else:
            # print(env.get_sql())
            # sqls.add(tuple(buffer.states.tolist()))
            if scount % 20 == 0 or tcount % 1000 == 0:
                print(scount, '----', tcount)
            end_of_pretrain_episode_actions(reward, ep_steps, buffer, memory, scount)
            scount += 1
            if scount % 64 == 0:
                cpath = os.path.abspath('.')
                tpath = cpath + '/' + dbname + '/' + env.task_name + '_predata.npy'
                np.save(tpath, memory)
            buffer.clear()
    cpath = os.path.abspath('.')
    tpath = cpath + '/' + dbname + '/' + env.task_name + '_predata.npy'
    np.save(tpath, memory)


def prc_predata():
    para = sys.argv
    dbname = para[1]
    mtype = para[2]  # point/range
    if mtype == 'point':
        # print('enter point')
        pc = int(para[3])
        nums = int(para[4])
        pre_data(dbname, mtype, pc, nums)
    elif mtype == 'range':
        rc = (int(para[3]), int(para[4]))
        nums = int(para[5])
        pre_data(dbname, mtype, rc, nums)
    else:
        print("error")


def random_generate(dbname, tpath, numbers):
    env = GenSqlEnv(metric=100000, dbname=dbname, target_type=0)
    episode = 0
    max_episodes = numbers
    f = open(tpath, 'w')
    while episode < max_episodes:
        # print('第', episode, '条')
        current_state = env.reset()
        reward, done = env.bug_reward, False
        ep_steps = 0
        while not (done or ep_steps >= SEQ_LENGTH):
            action = choose_action(env.observe(current_state))
            reward, done = env.step(action)
            ep_steps += 1
            current_state = action
        if ep_steps == SEQ_LENGTH or reward == env.bug_reward:
            print('采样忽略')
        else:
            episode += 1
            sql = env.get_sql()
            # print(sql)
            f.write(sql)
            # print('reward:', reward)
    f.close()


def test(dbname, numbers):
    env = GenSqlEnv(metric=100000, dbname=dbname, target_type=0)
    episode = 0
    max_episodes = numbers
    while episode < max_episodes:
        # print('第', episode, '条')
        current_state = env.reset()
        reward, done = env.bug_reward, False
        ep_steps = 0
        while not (done or ep_steps >= SEQ_LENGTH):
            action = choose_action(env.observe(current_state))
            reward, done = env.step(action)
            ep_steps += 1
            current_state = action
            # print(env.get_sql(), '', reward)
        if ep_steps == SEQ_LENGTH or reward == env.bug_reward:
            print('采样忽略')
        else:
            print(env.get_sql())
            res = env.get_cost()
            assert res != -1
            episode += 1
            # sql = env.get_sql()
            # print(sql)
            # print('reward:', reward)


if __name__ == '__main__':
    # cpath = os.path.abspath('.')
    # print('cur_path:', cpath)
    # numbers = 100
    # qpath = cpath + '/tpch/tpch'+str(numbers)
    # spath = qpath + '_result'
    # random_generate('tpch', qpath, numbers)
    prc_predata()
    # tpath = cpath + '/imdbload/imdbload10000'
    # random_generate('imdbload', tpath, 10000)
    # test('imdbload', 10000)












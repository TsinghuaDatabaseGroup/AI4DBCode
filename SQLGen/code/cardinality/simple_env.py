import numpy as np
from treelib import Tree
import math
import os
import pickle
import base
from cardinality.sample_data import MetaDataSupport
import sys

np.set_printoptions(threshold=np.inf)

# support grammar keyword
# 一些设定 #
# 聚合函数的出现一定会导致group by
# group by 一定程度出现having 对输出结果的控制
# operator = ['=', '!=', '>', '<', '<=', '>=']
operator = ['>', '<', '<=', '>=']
conjunction = ['and']
keyword = ['from', 'where']
# join = ['join']


class DataNode(object):
    # action_index 与 identifier不同, action_index是map里面的，identifire是tree里面的
    def __init__(self, action_index):
        self.action_index = action_index


DataType = base.DataType

# db, cursor = base.connect_server('tpch')


class GenSqlEnv(object):
    def __init__(self, metric, dbname, target_type, server_name='postgresql', allowed_error=0.1):
        self.target_type = target_type
        self.allowed_error = allowed_error
        self.metric = metric
        if target_type == 0:
            self.target = metric
            self.task_name = "card_pc{}".format(metric)
        else:
            self.low_b = metric[0]
            self.up_b = metric[1]
            self.task_name = "card_rc{}_{}".format(self.low_b, self.up_b)

        self.dbname = dbname
        self.server_name = server_name

        self.db, self.cursor = base.connect_server(dbname, server_name=server_name)
        self.SampleData = MetaDataSupport(dbname, metric)
        self.schema = self.SampleData.schema
        #  ....常量......  #
        self.step_reward = 0
        self.bug_reward = -100
        self.terminal_word = " "  # 空格结束、map在0上,index为0

        self.word_num_map, self.num_word_map, self.relation_tree = self._build_relation_env()
        self.relation_graph = base.build_relation_graph(self.dbname, self.schema)
        self.action_space = self.observation_space = len(self.word_num_map)

        # self.grammar_tree = self._build_grammar_env()
        self.from_space = []
        self.where_space = []

        self.operator = [self.word_num_map[x] for x in operator]
        self.conjunction = [self.word_num_map[x] for x in conjunction]
        self.keyword = [self.word_num_map[x] for x in keyword]
        # self.join = [self.word_num_map[x] for x in join]
        self.attributes = []

        table_node = self.relation_tree.children(self.relation_tree.root)
        self.tables = [field.identifier for field in table_node]
        for node in table_node:
             self.attributes += [field.identifier for field in self.relation_tree.children(node.identifier)]

        self.from_clause = self.where_clause = ""

        self.master_control = {
            'from': [self.from_observe, self.from_action],
            'where': [self.where_observe, self.where_action],
        }
        self.cur_state = self.master_control['from']  # 初始时为from
        self.time_step = 0

    def _build_relation_env(self):
        print("_build_env")

        schema = self.SampleData.schema
        sample_data = self.SampleData.get_data()

        tree = Tree()
        tree.create_node("root", 0, None, data=DataNode(0))

        word_num_map = dict()
        num_word_map = dict()

        word_num_map[self.terminal_word] = 0
        num_word_map[0] = self.terminal_word

        # 第一层 table_names
        count = 1
        for table_name in schema.keys():
            tree.create_node(table_name, count, parent=0, data=DataNode(count))
            word_num_map[table_name] = count
            num_word_map[count] = table_name
            count += 1

        # 第二层 table的attributes
        for table_name in schema.keys():
            for field in schema[table_name]:
                attribute = '{0}.{1}'.format(table_name, field)
                tree.create_node(attribute, count, parent=word_num_map[table_name],
                                 data=DataNode(count))
                word_num_map[attribute] = count
                num_word_map[count] = attribute
                count += 1

        # 第三层 每个table的sample data
        for table_name in schema.keys():
            for field in schema[table_name]:
                for data in sample_data[table_name][field]:
                    if data not in word_num_map.keys():
                        word_num_map[data] = len(num_word_map)
                        num_word_map[len(num_word_map)] = data
                    field_name = '{0}.{1}'.format(table_name, field)
                    tree.create_node(data, count, parent=word_num_map[field_name], data=DataNode(word_num_map[data]))
                    count += 1

        self.add_map(operator, word_num_map, num_word_map)
        self.add_map(conjunction, word_num_map, num_word_map)
        self.add_map(keyword, word_num_map, num_word_map)
        # self.add_map(join, word_num_map, num_word_map)

        print("_build_env done...")
        print("action/observation space:", len(num_word_map), len(word_num_map))
        print("relation tree size:", tree.size())
        # tree.show()
        return word_num_map, num_word_map, tree

    def reset(self):
        # print("reset")
        self.cur_state = self.master_control['from']
        self.from_clause = self.where_clause = ""
        self.where_space.clear()
        self.from_space.clear()
        self.time_step = 0
        return self.word_num_map['from']

    def activate_space(self, cur_space, keyword):   # 用keyword开启 cur_space 到 next_space 的门
        # 激活下一个space
        cur_space[keyword] = 1

    def activate_ternminal(self, cur_space):
        cur_space[0] = 1

    def from_observe(self, observation=None):
        if observation == self.word_num_map['from']:  # 第一次进来
            self.from_clause = 'from'
            candidate_tables = np.zeros((self.action_space,), dtype=int)
            candidate_tables[self.tables] = 1
            return candidate_tables
        else:  # observation in self.tables:   # 选择table 激活join type
            relation_tables = self.relation_graph.get_relation(self.num_word_map[observation])  # string类型
            relation_tables = set([self.word_num_map[table] for table in relation_tables])
            relation_tables = list(relation_tables.difference(self.from_space))  # 选过的不选了
            candidate_tables = np.zeros((self.action_space,), dtype=int)
            candidate_tables[relation_tables] = 1
            if len(self.from_space) > 0:
                candidate_tables[self.word_num_map['where']] = 1
                self.activate_ternminal(candidate_tables)
            return candidate_tables

    def from_action(self, action):
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
        elif action == self.word_num_map['where']:
            self.cur_state = self.master_control['where']
            self.cur_state[1](action)
        else:
            print('from error')
            # print(self.from_clause)
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
            return candidate_word
        elif observation in self.operator:
            candidate_word[self.operation_data(self.cur_attribtue)] = 1
            return candidate_word
        elif observation in self.conjunction:
            candidate_word[self.where_attributes] = 1
            return candidate_word
        else:   # data
            if len(self.where_attributes) != 0:
                candidate_word[self.conjunction] = 1
            self.activate_ternminal(candidate_word)
            return candidate_word

    def where_action(self, action):
        # print("enter where action")
        # print(self.num_word_map[action])
        if action == self.word_num_map['where']:
            self.where_clause = 'where '
        elif action in self.attributes:
            self.cur_attribtue = action
            self.where_clause = self.where_clause + self.num_word_map[action]
            self.where_attributes.remove(action)
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
            return final_reward, 1
        elif action == -1:
            return self.bug_reward, 1
        else:
            return self.cur_state[1](action)

    def get_sql(self):
        final_sql = 'select *'
        final_sql = final_sql + ' ' + self.from_clause
        if self.where_clause:
            final_sql = final_sql + ' ' + self.where_clause
        final_sql = final_sql + ';'
        return final_sql

    def cal_e_card(self):
        sql = self.get_sql()
        # print(sql)
        result, query_info = base.get_evaluate_query_info(self.dbname, sql)
        if result != 1:
            # print(sql)
            return -1
        return query_info['e_cardinality']

    def cal_reward(self):
        if self.target_type == 0:
            return self.cal_point_reward()
        else:
            return self.cal_range_reward()

    def is_satisfy(self):
        e_card = self.cal_e_card()
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

    def cal_point_reward(self):
        e_card = self.cal_e_card()
        if e_card == -1:
            return self.step_reward
        else:
            reward = (-base.relative_error(e_card, self.target) + self.allowed_error) * 10
            # if reward > 0:
            #     print(self.get_sql())
            #     print("e_card:{} reward:{}".format(e_card, reward))
            reward = max(reward, -1)
            return reward

    def cal_range_reward(self):
        e_card = self.cal_e_card()
        if e_card == -1:
            return self.step_reward
        else:
            # print(self.get_sql())
            if self.low_b <= e_card <= self.up_b:
                # print(self.get_sql())
                # print("e_card:{} reward:{}".format(e_card, 2))
                return 2
            else:
                relative_error = max(base.relative_error(e_card, self.up_b),
                                     base.relative_error(e_card, self.low_b))
                reward = -relative_error
                reward = max(reward, -2)
                # a = min(e_card / self.up_b, self.up_b / e_card)
                # b = min(e_card / self.low_b, self.low_b / e_card)
                # reward = max(a, b)
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
    env = GenSqlEnv(metric=(1000, 2000), dbname=dbname, target_type=1)
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
            # print(env.get_sql(), '', reward)
            ep_steps += 1
            current_state = action
        if ep_steps == SEQ_LENGTH or reward == env.bug_reward:
            print('采样忽略')
        else:
            episode += 1



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


# def pre_data(dbname, mtype, metric, nums):
#     # episode = np.hstack((self.buffer.states, discounted_ep_rs, self.buffer.actions))
#     # index = (self.episode - 1) % BATCH_SIZE
#     # self.memory[index, :] = episode
#     memory = np.zeros((nums, SEQ_LENGTH * 3))
#     buffer = base.Buffer()
#     if mtype == 'point':
#         env = GenSqlEnv(metric=metric, dbname=dbname, target_type=0)
#     elif mtype == 'range':
#         env = GenSqlEnv(metric=metric, dbname=dbname, target_type=1)
#     else:
#         print('error')
#         return
#     scount = 0
#     while scount < nums:
#         current_state = env.reset()
#         reward, done = env.bug_reward, False
#         ep_steps = 0
#         while not (done or ep_steps >= SEQ_LENGTH):
#             candidate_action = env.observe(current_state)
#             action = choose_action(candidate_action)
#             reward, done = env.step(action)
#             buffer.store(current_state, action, reward, ep_steps)  # 单步为0
#             ep_steps += 1
#             current_state = action
#         if ep_steps == SEQ_LENGTH or reward == env.bug_reward or reward < env.target_type - 0.2:
#             buffer.clear()  # 采样忽略
#             # print('采样忽略')
#         else:
#             end_of_pretrain_episode_actions(reward, ep_steps, buffer, memory, scount)
#             buffer.clear()
#             scount += 1
#             if scount % 100 == 0:
#                 cpath = os.path.abspath('.')
#                 tpath = cpath + '/' + dbname + '/' + env.task_name + '_predata.npy'
#                 np.save(tpath, memory)
#     # print(memory.dtype)
#     cpath = os.path.abspath('.')
#     tpath = cpath + '/' + dbname + '/' + env.task_name + '_predata.npy'
#     np.save(tpath, memory)
#     # c = np.load(tpath)
#     # print(c)
#

# def prc_predata():
#     para = sys.argv
#     dbname = para[1]
#     mtype = para[2]  # point/range
#     if mtype == 'point':
#         # print('enter point')
#         pc = int(para[3])
#         nums = int(para[4])
#         pre_data(dbname, mtype, pc, nums)
#     elif mtype == 'range':
#         rc = (int(para[3]), int(para[4]))
#         nums = int(para[5])
#         pre_data(dbname, mtype, rc, nums)
#     else:
#         print("error")


if __name__ == '__main__':
    random_generate('tpch', '/home/lixizhang/learnSQL/cardinality/tpch/tpch_random_10000', 10000)

















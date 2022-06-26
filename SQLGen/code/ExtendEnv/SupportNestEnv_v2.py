import numpy as np
import os
import base
import sys
from ExtendEnv.Common import Common
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


class HandleSubQuery(object):
    def __init__(self):
        self.relate_table = []
        self.relate_attribute = []
        self.alias = {}
        self.call_from = None

    def set_call_from(self, call_from):
        self.call_from = call_from

    def clear(self):
        self.relate_table = []
        self.relate_attribute = []
        self.call_from = None


class GenQueryEnv(object):
    def __init__(self, c_obj):
        self.c_obj = c_obj
        self.select_space = []
        self.from_space = []
        self.where_space = []
        self.group_by_space = []
        self.having_space = []
        self.order_by_space = []
        self.aggregate_space = []
        self.group_key = False
        self.is_alias = False
        self.handle_sub = HandleSubQuery()
        self.bug_reward = c_obj.bug_reward
        # print(self.relation_table)
        self.select_clause = self.from_clause = self.where_clause = self.group_by_clause = self.having_clause = \
            self.order_by_clause = self.aggregate_clause = self.subquery_clause = ""

        self.master_control = {
            'select': [self.select_observe, self.select_action],
            'from': [self.from_observe, self.from_action],
            'where': [self.where_observe, self.where_action],
            # 'group by': [self.group_by_observe, self.group_by_action],
            'having': [self.having_observe, self.having_action],
            'order by': [self.order_by_observe, self.order_by_action],
            'aggregate': [self.aggregate_observe, self.aggregate_action],
            'subquery': [self.subquery_observe, self.subquery_action],
        }

        self.cur_state = self.master_control['from']  # 初始时为from
        self.time_step = 0

    def transfer_attr(self, action):
        if self.is_alias:
            return "tmp1.{}".format(self.c_obj.num_word_map[action].split('.')[1])
        return self.c_obj.num_word_map[action]
        # return self.handle_sub.alias[self.c_obj.num_word_map[action]]

    def reset(self):
        # print("reset")
        self.cur_state = self.master_control['from']
        self.select_clause = self.from_clause = self.where_clause = self.group_by_clause = \
            self.having_clause = self.order_by_clause = self.aggregate_clause = self.subquery_clause = ""
        self.where_space.clear()
        self.from_space.clear()
        self.select_space.clear()
        self.aggregate_space.clear()
        self.group_by_space.clear()
        self.order_by_space.clear()
        self.having_space.clear()
        self.time_step = 0
        self.group_key = False
        self.is_alias = False
        return self.c_obj.word_num_map['from']

    def select_observe(self, observation):
        # self.cur_sql[: -1]
        candidate_word = np.zeros((self.c_obj.action_space,), dtype=int)
        # print('enter select observe')
        # print(self.c_obj.num_word_map[observation])
        if self.c_obj.num_word_map[observation] == 'select' or self.c_obj.num_word_map[observation] == '#':     # 第一次进
            self.need_select_table = self.from_space.copy()
            for table_index in self.need_select_table:
                candidate_word[[field.identifier for field in self.c_obj.relation_tree.children(table_index)]] = 1
            return candidate_word
        else:   # attribute
            if self.need_select_table:
                for table_index in self.need_select_table:
                    candidate_word[[field.identifier for field in self.c_obj.relation_tree.children(table_index)]] = 1
                return candidate_word
            else:   # table和普通的attribute选完了可以聚合也可以where condition 也可以orderby
                # self.c_obj.activate_space(candidate_word, 'aggregate')
                self.c_obj.activate_space(candidate_word, 'where')
                self.c_obj.activate_space(candidate_word, 'order by')
                self.c_obj.activate_terminal(candidate_word)
                return candidate_word

    def select_action(self, action):
        # print('enter select_action:', self.c_obj.num_word_map[action])
        if self.c_obj.num_word_map[action] == 'select':
            self.select_clause = 'select'
        elif action in self.c_obj.keyword:
            self.cur_state = self.master_control[self.c_obj.num_word_map[action]]
            self.cur_state[1](action)
        else:
            self.select_space.append(action)
            self.group_by_space.append(action)
            self.order_by_space.append(action)
            table_name_index = self.c_obj.relation_tree.parent(action).identifier   #
            self.need_select_table.remove(table_name_index)
            attr = self.transfer_attr(action)
            if self.select_clause == 'select':

                self.select_clause = self.select_clause + ' ' + attr
            else:
                self.select_clause = self.select_clause + ', ' + attr
        return self.c_obj.cal_reward(self.get_sql()), 0

    def aggregate_observe(self, observation=None):
        candidate_word = np.zeros((self.c_obj.action_space,), dtype=int)
        if self.group_key is False:
            self.group_by_generate()    # 直接group by产生
            self.group_key = True
        # self.c_obj.activate_space(candidate_word, self.c_obj.word_num_map['aggregate'])
        self.c_obj.activate_space(candidate_word, 'where')
        self.c_obj.activate_space(candidate_word, 'order by')
        self.c_obj.activate_space(candidate_word, 'having')
        self.c_obj.activate_terminal(candidate_word)
        return candidate_word

    def aggregate_action(self, action):
        if action == self.c_obj.word_num_map['aggregate']:
            table = np.random.choice(self.from_space)
            attributes = [node.identifier for node in self.c_obj.relation_tree.children(table)]
            choose_attribute = np.random.choice(attributes)
            choose_aggregate_type = np.random.choice(['max', 'min', 'avg', 'sum'])
            self.aggregate_space.append((choose_aggregate_type, choose_attribute))
            choose_attribute = self.transfer_attr(choose_attribute)
            self.aggregate_clause = self.aggregate_clause + ' ' + '{aggregate_type}({aggregate_attribute})'.format(
                aggregate_type=choose_aggregate_type, aggregate_attribute=choose_attribute)
        else:   # 其他key_word
            self.cur_state = self.master_control[self.c_obj.num_word_map[action]]
            self.cur_state[1](action)
        return self.c_obj.cal_reward(self.get_sql()), 0

    def from_observe(self, observation=None):
        if observation == self.c_obj.word_num_map['from']:    # 第一次进来
            candidate_word = np.zeros((self.c_obj.action_space,), dtype=int)
            # candidate_word[self.c_obj.tables] = 1
            self.c_obj.activate_space(candidate_word, 'subquery')
            return candidate_word
        else:  # observation in self.c_obj.tables:   # 选择table 激活join type
            relation_tables = self.c_obj.relation_graph.get_relation(self.c_obj.num_word_map[observation])  # string类型
            relation_tables = set([self.c_obj.word_num_map[table] for table in relation_tables])
            relation_tables = list(relation_tables.difference(self.from_space))     # 选过的不选了
            candidate_tables = np.zeros((self.c_obj.action_space,), dtype=int)
            candidate_tables[relation_tables] = 1
            if len(self.from_space) > 0:
                candidate_tables[self.c_obj.word_num_map['select']] = 1
            return candidate_tables

    def from_action(self, action):
        # print("enter from action")
        if action in self.c_obj.tables:
            self.from_space.append(action)
            if self.from_clause == 'from':
                self.from_clause = self.from_clause + ' ' + self.c_obj.num_word_map[self.from_space[0]]
            else:
                table1 = self.from_space[len(self.from_space)-1]
                table2 = self.from_space[len(self.from_space)-2]
                relation_key = self.c_obj.relation_graph.get_relation_key(self.c_obj.num_word_map[table1],
                                                                    self.c_obj.num_word_map[table2])
                f_relation = relation_key[0]
                t_relation = relation_key[1]
                join_condition = f_relation[0] + '=' + t_relation[0]
                for i in range(1, len(f_relation)):
                    join_condition = join_condition + ' and ' + f_relation[i] + '=' + t_relation[i]
                self.from_clause = self.from_clause + ' join ' + self.c_obj.num_word_map[table1] + ' on ' + join_condition
        elif action == self.c_obj.word_num_map['select']:
            self.cur_state = self.master_control['select']
            self.cur_state[1](action)
        elif action == self.c_obj.word_num_map['subquery']:
            self.is_alias = True
            self.handle_sub.set_call_from('from')
            self.cur_state = self.master_control['subquery']
            self.cur_state[1](action)
        elif action == self.c_obj.word_num_map['#']:
            self.from_clause = self.from_clause + " " + self.subquery_clause
            self.subquery_clause = ""
            self.cur_state = self.master_control['select']
            self.cur_state[1](self.c_obj.word_num_map['select'])
        elif action == self.c_obj.word_num_map['from']:
            self.from_clause = 'from'
        else:
            print('from error')
        return self.c_obj.cal_reward(self.get_sql()), 0

    def where_observe(self, observation):
        # print("enter where space")
        candidate_word = np.zeros((self.c_obj.action_space,), dtype=int)
        if observation == self.c_obj.word_num_map['where']:
            self.where_attributes = []
            for table_index in self.from_space:
                for field in self.c_obj.relation_tree.children(table_index):
                    self.where_attributes.append(field.identifier)
            candidate_word[self.where_attributes] = 1
            return candidate_word
        elif observation in self.c_obj.attributes:
            candidate_word[self.c_obj.operator] = 1
            # candidate_condition[self.predicate_type] = 1
            return candidate_word
        elif observation in self.c_obj.operator or observation in self.c_obj.predicate_in:
            # candidate_word[self.operation_data(self.cur_attribtue)] = 1
            self.c_obj.activate_space(candidate_word, 'subquery')
            return candidate_word
        elif observation in self.c_obj.conjunction:
            candidate_word[self.where_attributes] = 1
            return candidate_word
        else:   # data
            candidate_word[self.c_obj.conjunction] = 1
            self.c_obj.activate_terminal(candidate_word)
            self.c_obj.activate_space(candidate_word, 'order by')
            if self.group_key:
                self.c_obj.activate_space(candidate_word, 'having')
            return candidate_word

        # elif observation in self.predicate_type:

    def where_action(self, action):
        # print("enter where action")
        # print(self.c_obj.num_word_map[action])
        if action == self.c_obj.word_num_map['where']:
            self.where_clause = 'where '
        elif action in self.c_obj.attributes:
            self.cur_attribtue = action
            self.where_clause = self.where_clause + self.transfer_attr(action)
        elif action in self.c_obj.operator:
            self.where_clause = self.where_clause + ' ' + self.c_obj.num_word_map[action] + ' '
        elif action in self.c_obj.conjunction:
            self.where_clause = self.where_clause + ' {} '.format(self.c_obj.num_word_map[action])
        elif action == self.c_obj.word_num_map['subquery']:
            cur_attribute = self.c_obj.num_word_map[self.cur_attribtue]
            self.handle_sub.relate_attribute = cur_attribute
            self.handle_sub.relate_table = cur_attribute.split('.')[0]
            self.handle_sub.set_call_from('where')
            self.cur_state = self.master_control[self.c_obj.num_word_map[action]]
            self.cur_state[1](action)
        elif action in self.c_obj.keyword:
            self.cur_state = self.master_control[self.c_obj.num_word_map[action]]
            self.cur_state[1](action)
        elif action == self.c_obj.word_num_map[self.c_obj.sub_terminal_word]:
            self.where_clause = self.where_clause + self.subquery_clause
            self.subquery_clause = ""
        else:   # data or subquery 结束
            self.where_clause = self.where_clause + str(self.c_obj.num_word_map[action])
        return self.c_obj.cal_reward(self.get_sql()), 0

    def operation_data(self, attributes):
        data = [node.data.action_index for node in self.c_obj.relation_tree.children(attributes)]
        return data

    def group_by_generate(self):
        self.group_by_clause = 'group by'
        for attribute in self.group_by_space:
            attr = self.transfer_attr(attribute)
            self.group_by_clause = self.group_by_clause + ' ' + attr + ','
        self.group_by_clause = self.group_by_clause[: -1]

    def having_observe(self, observation):
        # self.having_space是聚合函数 + terminal
        candidate_word = np.zeros((self.c_obj.action_space,), dtype=int)
        cur_word = self.c_obj.num_word_map[observation]
        if cur_word == 'having':
            self.c_obj.activate_terminal(candidate_word)
            self.c_obj.activate_space(candidate_word, 'order by')
        else:
            print("error")
        return candidate_word

    def having_action(self, action):
        # print("having action:", action, "===", self.c_obj.num_word_map[action])
        if action == self.c_obj.word_num_map['having']:
            attr = self.transfer_attr(self.aggregate_space[0][1])
            agg_type = self.aggregate_space[0][0]
            chosen_op = np.random.choice(['=', '!=', '>', '<', '<=', '>='])
            chosen_data = np.random.choice(self.operation_data(self.aggregate_space[0][1]))
            self.having_clause = 'having {}({}) {} {}'.format(agg_type, attr, chosen_op, chosen_data)
        else:
            self.cur_state = self.master_control[self.c_obj.num_word_map[action]]
            self.cur_state[1](action)
        return self.c_obj.cal_reward(self.get_sql()), 0

    def order_by_observe(self, observation):
        candidate_word = np.zeros((self.c_obj.action_space,), dtype=int)
        candidate_word[self.select_space] = 1
        if observation != self.c_obj.word_num_map['order by']:
            self.c_obj.activate_terminal(candidate_word)
        return candidate_word

    def order_by_action(self, action):
        if action == self.c_obj.word_num_map['order by']:
            self.order_by_clause = 'order by'
        else:
            self.select_space.remove(action)
            choose_order = np.random.choice(['DESC', 'ASC'])
            attr = self.transfer_attr(action)
            if self.order_by_clause == 'order by':
                self.order_by_clause = self.order_by_clause + ' ' + attr + ' ' + choose_order
            else:
                self.order_by_clause = self.order_by_clause + ', ' + attr + ' ' + choose_order
        return self.c_obj.cal_reward(self.get_sql()), 0

    def subquery_observe(self, observation):
        func = eval("self.subquery_from_{}_observe".format(self.handle_sub.call_from))
        return func(observation)

    def subquery_action(self, action):
        func = eval("self.subquery_from_{}_action".format(self.handle_sub.call_from))
        return func(action)

    def subquery_from_where_observe(self, observation):
        candidate_word = np.zeros((self.c_obj.action_space,), dtype=int)
        if observation == self.c_obj.word_num_map['subquery']:
            self.subquery_attributes = []
            for field in self.c_obj.relation_tree.children(self.c_obj.word_num_map[self.handle_sub.relate_table]):
                self.subquery_attributes.append(field.identifier)
            candidate_word[self.subquery_attributes] = 1
            # self.c_obj.activate_space(candidate_word, self.c_obj.word_num_map[self.terminal_word])
        elif observation in self.c_obj.attributes:
            candidate_word[self.c_obj.operator] = 1
        elif observation in self.c_obj.operator:
            candidate_word[self.operation_data(self.cur_attribtue)] = 1
        elif observation in self.c_obj.conjunction:
            candidate_word[self.subquery_attributes] = 1
        else:  # data
            candidate_word[self.c_obj.conjunction] = 1
            self.c_obj.activate_space(candidate_word, self.c_obj.sub_terminal_word)
        return candidate_word

    def subquery_from_where_action(self, action):
        if action == self.c_obj.word_num_map['subquery']:
            self.subquery_clause = "select {} from {} where".format(self.handle_sub.relate_attribute,
                                                                    self.handle_sub.relate_table)
        elif action == self.c_obj.word_num_map[self.c_obj.sub_terminal_word]:
            self.subquery_clause = "({})".format(self.subquery_clause)
            self.cur_state = self.master_control['where']
            self.cur_state[1](action)
        elif action in self.c_obj.attributes:
            self.cur_attribtue = action
            self.subquery_clause = self.subquery_clause + " " + self.transfer_attr(action)
        elif action in self.c_obj.operator or action in self.c_obj.conjunction:
            self.subquery_clause = self.subquery_clause + ' ' + self.c_obj.num_word_map[action] + ' '
        else:
            self.subquery_clause = self.subquery_clause + str(self.c_obj.num_word_map[action])
        return self.c_obj.cal_reward(self.get_sql()), 0

    def subquery_from_from_observe(self, observation):
        candidate_word = np.zeros((self.c_obj.action_space,), dtype=int)
        if observation == self.c_obj.word_num_map['subquery']:
            candidate_word[self.c_obj.tables] = 1
        elif observation in self.c_obj.tables:
            relation_tables = self.c_obj.relation_graph.get_relation(self.c_obj.num_word_map[observation])
            relation_tables = set([self.c_obj.word_num_map[table] for table in relation_tables])
            relation_tables = list(relation_tables.difference(self.from_space))
            candidate_word[relation_tables] = 1
            self.c_obj.activate_space(candidate_word, self.c_obj.sub_terminal_word)
        else:
            print(self.c_obj.num_word_map[observation])
            print('error')
        return candidate_word

    def subquery_from_from_action(self, action):
        if action == self.c_obj.word_num_map['subquery']:
            pass
        elif action == self.c_obj.word_num_map[self.c_obj.sub_terminal_word]:
            from_tables = [self.c_obj.num_word_map[x] for x in self.from_space]
            self.subquery_clause = "(select {} from {}) as tmp1".format('*', ', '.join(from_tables))
            # print (self.subquery_clause)
            self.cur_state = self.master_control['from']
            self.cur_state[1](action)
        elif action in self.c_obj.tables:
            self.from_space.append(action)
        else:
            print(action)
            print('error')
            exit(0)
        return self.c_obj.cal_reward(self.get_sql()), 0

    def observe(self, observation):
        """
        :param observation: index 就可以
        :return: 返回vocabulary_size的矩阵，单步reward
        """
        if observation == self.c_obj.word_num_map['query']:
            return self.cur_state[0](self.c_obj.word_num_map['from'])
        return self.cur_state[0](observation)

    def step(self, action):
        self.time_step += 1
        if action == self.c_obj.word_num_map[self.c_obj.terminal_word]:  # choose 结束：
            # return self.final_reward(), 1
            final_reward = self.c_obj.cal_reward(self.get_sql())
            # assert final_reward != 0
            return final_reward, 1
        elif action == -1:
            return self.bug_reward, 1
        elif action == self.c_obj.word_num_map['query']:
            return self.cur_state[1](self.c_obj.word_num_map['from'])
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


def choose_action(observation):
    candidate_list = np.argwhere(observation == np.max(observation)).flatten()
    # action = np.random.choice(candidate_list, p=increase_key_probability(candidate_list, key_word_list, step))
    action = np.random.choice(candidate_list)
    return action


SEQ_LENGTH = 40


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


def test(dbname, numbers):
    c_obj = Common(metric=100000, dbname=dbname, target_type=0)
    env = GenQueryEnv(c_obj)
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
            sql = env.get_sql()
            print(sql)
            res, info = base.get_evaluate_query_info('tpch', sql)
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
    # prc_predata()
    # tpath = cpath + '/imdbload/imdbload10000'
    # random_generate('imdbload', tpath, 10000)
    test('tpch', 10000)












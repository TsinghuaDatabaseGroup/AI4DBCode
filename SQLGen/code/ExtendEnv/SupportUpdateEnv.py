from ExtendEnv.Common import Common
import base
import numpy as np


class GenUpdateEnv(object):
    def __init__(self, c_obj):
        self.c_obj = c_obj
        self.sql = ""
        self.from_clause = ""
        self.cur_table = ""
        self.set_clause = ""
        self.where_clause = ""
        self.subquery_clause = ""
        self.master_control = {
            'from': [self.from_observe, self.from_action],
            'where': [self.where_observe, self.where_action],
            'subquery': [self.subquery_observe, self.subquery_action],
            'set': [self.set_observe, self.set_action]
        }

        self.cur_state = self.master_control['from']  # 初始时为from
        self.time_step = 0
        self.bug_reward = c_obj.bug_reward

    def reset(self):
        """
        :return:
        """
        self.cur_table = ""
        self.where_clause = ""
        self.subquery_clause = ""
        self.from_clause = ""
        self.set_clause = ""
        self.sql = ""
        self.cur_state = self.master_control['from']
        return self.c_obj.word_num_map['from']

    def get_sql(self):
        """
        :return:
        """
        self.sql = "update {} set {}".format(self.from_clause, self.set_clause)
        if self.where_clause:
            self.sql = self.sql + ' ' + self.where_clause
        return self.sql

    def operation_data(self, attributes):
        data = [node.data.action_index for node in self.c_obj.relation_tree.children(attributes)]
        return data

    def from_observe(self, observation=None):
        candidate_word = np.zeros((self.c_obj.action_space,), dtype=int)
        if observation == self.c_obj.word_num_map['from']:  # 第一次进来
            candidate_word[self.c_obj.tables] = 1
        else:  # observation in self.c_obj.tables:
            self.c_obj.activate_space(candidate_word, 'set')
        return candidate_word

    def from_action(self, action):
        # print("enter from action")
        # print(self.c_obj.num_word_map[action])
        if action in self.c_obj.tables:
            self.cur_table = action
            self.from_clause = self.c_obj.num_word_map[action]
        elif action == self.c_obj.word_num_map['from']:
            pass
        elif action in self.c_obj.keyword:
            self.cur_state = self.master_control[self.c_obj.num_word_map[action]]
            self.cur_state[1](action)
        else:
            exit('from error')
        return self.c_obj.cal_reward(self.get_sql()), 0

    def set_observe(self, observation):
        """
        :param observation:
        :return:
        """
        candidate_word = np.zeros((self.c_obj.action_space,), dtype=int)
        if observation == self.c_obj.word_num_map['set']:
            self.set_attribute = []
            for field in self.c_obj.relation_tree.children(self.cur_table):
                self.set_attribute.append(field.identifier)
            # print(self.set_attribute)
            candidate_word[self.set_attribute] = 1
        elif observation in self.c_obj.attributes:
            candidate_word[self.operation_data(observation)] = 1
        else:
            self.c_obj.activate_space(candidate_word, 'where')
            self.c_obj.activate_terminal(candidate_word)
            candidate_word[self.set_attribute] = 1
        return candidate_word

    def set_action(self, action):
        """
        :param action:
        :return:
        """
        if action == self.c_obj.word_num_map['set']:
            pass
        elif action in self.c_obj.attributes:
            attr = self.c_obj.num_word_map[action].split('.')[1]
            if self.set_clause == "":
                self.set_clause = attr + " = "
            else:
                self.set_clause = self.set_clause + ", " + attr + " = "
            self.set_attribute.remove(action)
        elif action in self.c_obj.keyword:
            self.cur_state = self.master_control[self.c_obj.num_word_map[action]]
            self.cur_state[1](action)
        else:
            self.set_clause = self.set_clause + str(self.c_obj.num_word_map[action])
        return self.c_obj.cal_reward(self.get_sql()), 0

    def where_observe(self, observation):
        """
        :return:
        """
        candidate_word = np.zeros((self.c_obj.action_space,), dtype=int)
        if observation == self.c_obj.word_num_map['where']:
            self.where_attributes = []
            for field in self.c_obj.relation_tree.children(self.cur_table):
                self.where_attributes.append(field.identifier)
            candidate_word[self.where_attributes] = 1
            return candidate_word
        elif observation in self.c_obj.attributes:
            candidate_word[self.c_obj.operator] = 1
            return candidate_word
        elif observation in self.c_obj.operator or observation in self.c_obj.predicate_in:
            candidate_word[self.operation_data(self.cur_attribute)] = 1
            self.c_obj.activate_space(candidate_word, 'subquery')
            return candidate_word
        elif observation in self.c_obj.conjunction:
            candidate_word[self.where_attributes] = 1
            return candidate_word
        else:  # data
            candidate_word[self.c_obj.conjunction] = 1
            self.c_obj.activate_terminal(candidate_word)
            return candidate_word

    def where_action(self, action):
        """
        :return:
        """
        # print("enter where action")
        # print(self.c_obj.num_word_map[action])
        if action == self.c_obj.word_num_map['where']:
            self.where_clause = 'where '
        elif action in self.c_obj.attributes:
            self.cur_attribute = action
            self.where_clause = self.where_clause + self.c_obj.num_word_map[action]
        elif action in self.c_obj.operator:
            self.where_clause = self.where_clause + ' ' + self.c_obj.num_word_map[action] + ' '
        elif action in self.c_obj.conjunction:
            self.where_clause = self.where_clause + ' {} '.format(self.c_obj.num_word_map[action])
        elif action in self.c_obj.keyword:
            self.cur_state = self.master_control[self.c_obj.num_word_map[action]]
            self.cur_state[1](action)
        elif action == self.c_obj.word_num_map[self.c_obj.sub_terminal_word]:
            self.where_clause = self.where_clause + self.subquery_clause
            self.subquery_clause = ""
        else:  # data or subquery 结束
            self.where_clause = self.where_clause + str(self.c_obj.num_word_map[action])
        return self.c_obj.cal_reward(self.get_sql()), 0

    def subquery_observe(self, observation):
        """
        :return:
        """
        candidate_word = np.zeros((self.c_obj.action_space,), dtype=int)
        if observation == self.c_obj.word_num_map['subquery']:
            self.subquery_attributes = []
            for field in self.c_obj.relation_tree.children(self.cur_table):
                self.subquery_attributes.append(field.identifier)
            candidate_word[self.subquery_attributes] = 1
            # self.c_obj.activate_space(candidate_word, self.c_obj.word_num_map[self.c_obj.terminal_word])
        elif observation in self.c_obj.attributes:
            candidate_word[self.c_obj.operator] = 1
        elif observation in self.c_obj.operator:
            candidate_word[self.operation_data(self.cur_attribute)] = 1
        elif observation in self.c_obj.conjunction:
            candidate_word[self.subquery_attributes] = 1
        else:  # data
            candidate_word[self.c_obj.conjunction] = 1
            self.c_obj.activate_space(candidate_word, self.c_obj.word_num_map[self.c_obj.sub_terminal_word])
        return candidate_word

    def subquery_action(self, action):
        """
        :return:
        """
        # print('sub action')
        # print(self.c_obj.num_word_map[action])
        if action == self.c_obj.word_num_map['subquery']:
            self.subquery_clause = "select {} from {} where".format(self.c_obj.num_word_map[self.cur_attribute],
                                                                    self.c_obj.num_word_map[self.cur_table])
        elif action == self.c_obj.word_num_map[self.c_obj.sub_terminal_word]:
            self.subquery_clause = "({})".format(self.subquery_clause)
            self.cur_state = self.master_control['where']
            self.cur_state[1](action)
        elif action in self.c_obj.attributes:
            self.cur_attribute = action
            # print('sub attr', self.c_obj.num_word_map[action])
            self.subquery_clause = self.subquery_clause + ' ' + self.c_obj.num_word_map[action]
        elif action in self.c_obj.operator or action in self.c_obj.conjunction:
            self.subquery_clause = self.subquery_clause + ' ' + self.c_obj.num_word_map[action] + ' '
        else:
            self.subquery_clause = self.subquery_clause + str(self.c_obj.num_word_map[action])
        return self.c_obj.cal_reward(self.get_sql()), 0

    def observe(self, observation):
        if observation == self.c_obj.word_num_map['update']:
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
        elif action == self.c_obj.word_num_map['update']:
            return self.cur_state[1](self.c_obj.word_num_map['from'])
        else:
            return self.cur_state[1](action)


def choose_action(observation):
    candidate_list = np.argwhere(observation == np.max(observation)).flatten()
    # action = np.random.choice(candidate_list, p=increase_key_probability(candidate_list, key_word_list, step))
    action = np.random.choice(candidate_list)
    return action


SEQ_LENGTH = 30


def test(dbname, numbers):
    c_obj = Common(metric=100000, dbname=dbname, target_type=0)
    env = GenUpdateEnv(c_obj=c_obj)
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
            assert res != 0
            episode += 1
            # print('reward:', reward)


if __name__ == '__main__':
    test('tpch', 100000)
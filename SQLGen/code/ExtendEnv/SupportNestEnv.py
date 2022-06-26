import numpy as np
import base
from ExtendEnv.Common import Common
np.set_printoptions(threshold=np.inf)

# support grammar keyword
# 一些设定 #
# 聚合函数的出现一定会导致group by
# group by 一定程度出现having 对输出结果的控制
# operator = ['=', '!=', '>', '<', '<=', '>=']
operator = ['>', '<', '<=', '>=', '<>']
predicate_in = ['in', 'not in']
conjunction = ['and']
keyword = ['from', 'where']
# join = ['join']


class DataNode(object):
    # action_index 与 identifier不同, action_index是map里面的，identifire是tree里面的
    def __init__(self, action_index):
        self.action_index = action_index


DataType = base.DataType

# db, cursor = base.connect_server('tpch')


class GenQueryEnv(object):
    def __init__(self, c_obj):
        self.c_obj = c_obj
        self.from_space = []
        self.where_space = []
        self.from_clause = self.where_clause = ""

        self.master_control = {
            'from': [self.from_observe, self.from_action],
            'where': [self.where_observe, self.where_action],
        }
        self.cur_state = self.master_control['from']  # 初始时为from
        self.time_step = 0
        self.bug_reward = c_obj.bug_reward

    def reset(self):
        # print("reset")
        self.cur_state = self.master_control['from']
        self.from_clause = self.where_clause = ""
        self.where_space.clear()
        self.from_space.clear()
        self.time_step = 0
        return self.c_obj.word_num_map['from']

    def from_observe(self, observation=None):
        if observation == self.c_obj.word_num_map['from']:  # 第一次进来
            self.from_clause = 'from'
            candidate_tables = np.zeros((self.c_obj.action_space,), dtype=int)
            candidate_tables[self.c_obj.tables] = 1
            return candidate_tables
        else:  # observation in self.c_obj.tables:   # 选择table 激活join type
            relation_tables = self.c_obj.relation_graph.get_relation(self.c_obj.num_word_map[observation])  # string类型
            relation_tables = set([self.c_obj.word_num_map[table] for table in relation_tables])
            relation_tables = list(relation_tables.difference(self.from_space))  # 选过的不选了
            candidate_tables = np.zeros((self.c_obj.action_space,), dtype=int)
            candidate_tables[relation_tables] = 1
            if len(self.from_space) > 0:
                candidate_tables[self.c_obj.word_num_map['where']] = 1
                self.c_obj.activate_terminal(candidate_tables)
            return candidate_tables

    def from_action(self, action):
        if action in self.c_obj.tables:
            self.from_space.append(action)
            if self.from_clause == 'from':
                self.from_clause = self.from_clause + ' ' + self.c_obj.num_word_map[self.from_space[0]]
            else:
                table1 = self.from_space[len(self.from_space)-1]
                table2 = self.from_space[len(self.from_space)-2]
                relation_key = self.c_obj.relation_graph.get_relation_key(self.c_obj.num_word_map[table1],
                                                                    self.c_obj.num_word_map[table2])
                frelation = relation_key[0]
                trelation = relation_key[1]
                join_condition = frelation[0] + '=' + trelation[0]
                for i in range(1, len(frelation)):
                    join_condition = join_condition + ' and ' + frelation[i] + '=' + trelation[i]
                self.from_clause = self.from_clause + ' join ' + self.c_obj.num_word_map[table1] + ' on ' + join_condition
        elif action == self.c_obj.word_num_map['from']:
            pass
        elif action == self.c_obj.word_num_map['where']:
            self.cur_state = self.master_control['where']
            self.cur_state[1](action)
        else:
            print('from error')
            # print(self.from_clause)
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
            # candidate_word[self.c_obj.predicate_in] = 1
            return candidate_word
        elif observation in self.c_obj.operator or observation in self.c_obj.predicate_in:
            candidate_word[self.operation_data(self.cur_attribute)] = 1
            return candidate_word
        elif observation in self.c_obj.conjunction:
            candidate_word[self.where_attributes] = 1
            return candidate_word
        else:   # data
            if len(self.where_attributes) != 0:
                candidate_word[self.c_obj.conjunction] = 1
            self.c_obj.activate_terminal(candidate_word)
            return candidate_word

    def where_action(self, action):
        # print("enter where action")
        # print(self.c_obj.num_word_map[action])
        if action == self.c_obj.word_num_map['where']:
            self.where_clause = 'where '
        elif action in self.c_obj.attributes:
            self.cur_attribute = action
            self.where_clause = self.where_clause + self.c_obj.num_word_map[action]
            self.where_attributes.remove(action)
        elif action in self.c_obj.operator or action in self.c_obj.predicate_in:
            self.where_clause = self.where_clause + ' ' + self.c_obj.num_word_map[action] + ' '
        elif action in self.c_obj.conjunction:
            self.where_clause = self.where_clause + ' {} '.format(self.c_obj.num_word_map[action])
        elif action in self.c_obj.keyword:
            self.cur_state = self.master_control[self.c_obj.num_word_map[action]]
            self.cur_state[1](action)
        else:   # data
            self.where_clause = self.where_clause + str(self.c_obj.num_word_map[action])
        return self.c_obj.cal_reward(self.get_sql()), 0

    def operation_data(self, attributes):
        data = [node.data.action_index for node in self.c_obj.relation_tree.children(attributes)]
        return data


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
        final_sql = 'select *'
        final_sql = final_sql + ' ' + self.from_clause
        if self.where_clause:
            final_sql = final_sql + ' ' + self.where_clause
        final_sql = final_sql + ';'
        return final_sql


def choose_action(observation):
    candidate_list = np.argwhere(observation == np.max(observation)).flatten()
    # action = np.random.choice(candidate_list, p=increase_key_probability(candidate_list, key_word_list, step))
    action = np.random.choice(candidate_list)
    return action


SEQ_LENGTH = 40


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

















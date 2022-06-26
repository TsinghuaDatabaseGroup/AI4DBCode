from ExtendEnv.SupportDeleteEnv import GenDeleteEnv
from ExtendEnv.SupportInsertEnv import GenInsertEnv
from ExtendEnv.SupportUpdateEnv import GenUpdateEnv
from ExtendEnv.SupportNestEnv import GenQueryEnv
from ExtendEnv.Common_card import Common
import numpy as np
import base
import os
import sys


class Env(object):
    def __init__(self, metric, dbname, target_type):
        self.c_obj = Common(metric, dbname, target_type)
        self.QueryEnv = GenQueryEnv(self.c_obj)
        self.InsertEnv = GenInsertEnv(self.c_obj)
        self.UpdateEnv = GenUpdateEnv(self.c_obj)
        self.DeleteEnv = GenDeleteEnv(self.c_obj)

        self.Map = {
            'query': self.QueryEnv,
            'insert': self.InsertEnv,
            'update': self.UpdateEnv,
            'delete': self.DeleteEnv
        }
        self.dbname = dbname
        self.observation_space = self.action_space = self.c_obj.action_space
        self.state = self.c_obj.start_word
        self.bug_reward = self.c_obj.bug_reward
        self.max_length = self.c_obj.SEQ_LENGTH
        self.task_name = self.c_obj.task_name

    def reset(self):
        if self.state in self.Map:
            self.Map[self.state].reset()
        self.state = self.c_obj.start_word
        return self.c_obj.word_num_map[self.c_obj.start_word]

    def observe(self, observation):
        if observation == self.c_obj.word_num_map[self.c_obj.start_word]:
            candidate_word = np.zeros((self.observation_space,), dtype=int)
            self.c_obj.activate_space(candidate_word, 'query')
            self.c_obj.activate_space(candidate_word, 'insert')
            self.c_obj.activate_space(candidate_word, 'update')
            self.c_obj.activate_space(candidate_word, 'delete')
            return candidate_word
        else:
            return self.Map[self.state].observe(observation)

    def step(self, action):
        if self.c_obj.num_word_map[action] in self.Map:
            self.state = self.c_obj.num_word_map[action]
        return self.Map[self.state].step(action)

    def get_sql(self):
        return self.Map[self.state].get_sql()

    def is_satisfy(self, sql):
        return self.c_obj.is_satisfy(sql)


def choose_action(observation):
    candidate_list = np.argwhere(observation == np.max(observation)).flatten()
    # action = np.random.choice(candidate_list, p=increase_key_probability(candidate_list, key_word_list, step))
    action = np.random.choice(candidate_list)
    return action


def test(dbname, numbers):
    env = Env(metric=100000, dbname=dbname, target_type=0)
    episode = 0
    max_episodes = numbers
    while episode < max_episodes:
        current_state = env.reset()
        reward, done = env.bug_reward, False
        ep_steps = 0
        while not (done or ep_steps >= env.max_length):
            # print(current_state)
            action = choose_action(env.observe(current_state))
            reward, done = env.step(action)
            ep_steps += 1
            current_state = action
            # print(env.get_sql(), '', reward)
        if ep_steps == env.max_length or reward == env.bug_reward:
            if reward == env.bug_reward:
                print(env.get_sql())
                exit(0)
            print('Over Max Length')
        else:
            # print(env.get_sql())
            # res = env.get_cost()
            # assert res != -1
            episode += 1
            sql = env.get_sql()
            print(sql)
            print('reward:', reward)


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


def pre_data(dbname, mtype, metric, nums):
    # episode = np.hstack((self.buffer.states, discounted_ep_rs, self.buffer.actions))
    # index = (self.episode - 1) % BATCH_SIZE
    # self.memory[index, :] = episode
    memory = np.zeros((nums, SEQ_LENGTH * 3))
    buffer = base.Buffer()
    if mtype == 'point':
        env = Env(metric=metric, dbname=dbname, target_type=0)
    elif mtype == 'range':
        env = Env(metric=metric, dbname=dbname, target_type=1)
    else:
        print('error type')
        return
    tcount = 0
    scount = 0
    sqls = list()
    cpath = os.path.abspath('.')
    tpath = cpath + '/' + dbname + '/' + env.task_name + '_predata.npy'
    spath = cpath + '/' + dbname + '/' + env.task_name + '_predata.sql'
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
            buffer.clear()
            # print('采样忽略')
        else:
            sqls.append(env.get_sql())
            # sqls.add(tuple(buffer.states.tolist()))
            if scount % 20 == 0 or tcount % 1000 == 0:
                print(scount, '----', tcount)
            end_of_pretrain_episode_actions(reward, ep_steps, buffer, memory, scount)
            scount += 1
            if scount % 64 == 0:
                np.save(tpath, memory)
                store_sql(spath, sqls)
            buffer.clear()
    store_sql(spath, sqls)
    np.save(tpath, memory)


def store_sql(path, sqls):
    with open(path, 'w') as f:
        for sql in sqls:
            f.write(sql+';\n')

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
    # test('tpch', 100000)
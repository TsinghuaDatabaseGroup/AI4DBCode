import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from matplotlib import pyplot as plt
from ExtendEnv.Env import Env as envCost
from ExtendEnv.Env_card import Env as envCard
import multiprocessing
import os
import time
from datetime import datetime
from pathlib import Path
import base
import sys
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')


########
workers_num = multiprocessing.cpu_count()
# workers_num = 2
EMB_DIM = 200   # embedding dimension
HIDDEN_DIM = 128    # hidden state dimension of lstm cell
SEQ_LENGTH = 40     # sequence length
SEED = 88
BATCH_SIZE = 128
GAMMA = 1    # reward discount
ENTROPY_BETA = 0     # should be a positive number
########


#%%
class Target_Lstm(keras.Model):
    # 策略网络
    def __init__(self, input_dim, out_dim):
        super(Target_Lstm, self).__init__()
        # 两层
        self.state0 = [tf.zeros([BATCH_SIZE, HIDDEN_DIM]), tf.zeros([BATCH_SIZE, HIDDEN_DIM])]
        self.state1 = [tf.zeros([BATCH_SIZE, HIDDEN_DIM]), tf.zeros([BATCH_SIZE, HIDDEN_DIM])]

        self.cur_state0 = [tf.zeros([BATCH_SIZE, HIDDEN_DIM]), tf.zeros([BATCH_SIZE, HIDDEN_DIM])]
        self.cur_state1 = [tf.zeros([BATCH_SIZE, HIDDEN_DIM]), tf.zeros([BATCH_SIZE, HIDDEN_DIM])]
        self.time_step = 0
        self.embedding = layers.Embedding(input_dim, EMB_DIM, input_length=SEQ_LENGTH)

        # self.rnn_cell0 = layers.LSTMCell(HIDDEN_DIM, dropout=0.5)
        # self.rnn_cell1 = layers.LSTMCell(HIDDEN_DIM, dropout=0.5)
        self.rnn_cell0 = layers.LSTMCell(HIDDEN_DIM)
        self.rnn_cell1 = layers.LSTMCell(HIDDEN_DIM)
        self.outlayer = layers.Dense(out_dim, activation='relu')

    def time_step_output(self, inputs, time_step):
        assert time_step == self.time_step
        # inputs: [None, 1, action]
        x = inputs
        # [None, 1, action] -> [None, 1, emb_dim]
        x = self.embedding(x)[:, 0, :]
        out0, self.cur_state0 = self.rnn_cell0(x, self.cur_state0)
        out1, self.cur_state1 = self.rnn_cell1(out0, self.cur_state1)
        out = self.outlayer(out1)   # [1, total_words]
        self.time_step += 1
        return out

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.embedding(x)
        state0 = self.state0
        state1 = self.state1
        out_put = []

        for word in tf.unstack(x, axis=1):
            out0, state0 = self.rnn_cell0(word, state0)
            out1, state1 = self.rnn_cell1(out0, state1)
            out_final = self.outlayer(out1)
            out_put.append(out_final)  # [seq_len, batchsz, total_words]

        out_put = tf.transpose(out_put, perm=[1, 0, 2])
        # print('target nn shape', out_put.shape)
        return out_put


class Buffer:
    def __init__(self):
        self.states = np.zeros(SEQ_LENGTH, dtype=int)
        self.actions = np.zeros(SEQ_LENGTH, dtype=int)
        self.rewards = np.zeros(SEQ_LENGTH, dtype=float)

    def store(self, state, action, reward, time_step):
        self.states[time_step] = state
        self.actions[time_step] = action
        self.rewards[time_step] = reward

    def clear(self):
        self.states[:] = 0
        self.actions[:] = 0
        self.rewards[:] = 0


#%%
class REINFORCE(keras.Model):
    def get_config(self):
        pass

    def __init__(self, env):
        super(REINFORCE, self).__init__()
        self.env = env
        self.state_size = env.observation_space
        self.action_size = env.action_space
        self.policy = Target_Lstm(self.state_size, self.action_size)
        self.policy_optimizer = optimizers.Adam(0.1)

        self.memory = np.zeros((BATCH_SIZE, SEQ_LENGTH * 3))
        self.buffer = Buffer()

        # path to save model
        self.model_name = env.task_name
        self.save_path = os.path.join(Path(__file__).parent, self.env.dbname, 'Models', self.model_name)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        else:
            # self.load_ac(self.save_path)
            print('no load')

        self.pretrain_episode = 0
        # self.max_pretrain_episodes = 6400
        self.pre_memory = self.load_predata()

        self.episode = 0
        self.max_episodes = BATCH_SIZE * 100

        self.episode_rewards = []
        self.aggregate_episode_rewards = {'episode': [], 'avg_rewards': [], 'max_rewards': [], 'min_rewards': []}
        self.save_training_plot = True

        # 统计
        self.aggregate_stats_window = BATCH_SIZE
        self.show_stats_interval = BATCH_SIZE
        self.save_models_interval = BATCH_SIZE * 5

        self.start_time = datetime.now()
        self.last_ep_start_time = datetime.now()

        # tensorboard relation
        self.log_dir = os.path.join(Path(__file__).parent, 'Logs', self.model_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.output_graph = False
        if self.output_graph:
            self.summary_writer = tf.summary.create_file_writer(self.log_dir)
            self.summary_writer.set_as_default()

        self.sqls = list()
        self.ac_log = False
        self.time_log = False
        cpath = os.path.abspath('.')
        tpath = cpath + '/' + self.env.dbname + '/' + self.env.task_name
        if self.ac_log:
            print('cal_ac...')
            self.ac_path = tpath + '_ac'
            self.curve_path = tpath + '_curve'
            self.accuracy = open(self.ac_path, 'w')
            self.curves = open(self.curve_path, 'w')
        if self.time_log:
            print('cal_time...')
            time.sleep(5)
            sql_path = tpath + '_sql'
            t_path = tpath + '_time'
            self.efficiency = open(t_path, 'w')
            self.sql_record = open(sql_path, 'w')

    def load_predata(self):
        cpath = os.path.abspath('.')
        tpath = cpath + '/' + self.env.dbname + '/' + self.env.task_name + '_predata.npy'
        pre_memory = np.load(tpath, allow_pickle=True)
        print("pre_data load down...")
        time.sleep(1)
        return pre_memory

    def select_action(self, s, candidate_action, time_step):
        o_t = self.policy.time_step_output(s, time_step)[0]  # [b, total_words]
        mask_o_t = tf.where(candidate_action, o_t, 0)     # acts as a mask
        mask_o_t = tf.constant(mask_o_t)[None, :]
        a = tf.random.categorical(mask_o_t, 1)[0]
        a = int(a)
        # 这里取交集可能为空, 在于初始化参数有的word本身概率为0
        if a == len(candidate_action) or candidate_action[a] == 0:  # 交集为空，实际a不可选, 这里有个api的bug，tensorflow2.0会返回len，tensorflow2.0-gpu会返回0
            a = self.random_select_action(candidate_action)
        return a

    def random_select_action(self, candidate_action):
        candidate_list = np.argwhere(candidate_action == np.max(candidate_action)).flatten()
        action = np.random.choice(candidate_list)
        return action

    def inference_select_action(self, s, candidate_action, time_step):
        o_t = self.policy.time_step_output(s, time_step)[0]  # [b, total_words]
        mask_o_t = tf.where(candidate_action, o_t, 0)  # acts as a mask
        mask_o_t_c = tf.constant(mask_o_t)[None, :]
        a = tf.random.categorical(mask_o_t_c, 1)[0]
        a = int(a)
        print("choose:{}, prob:{}".format(a, mask_o_t[a]))
        if a == len(candidate_action) or candidate_action[a] == 0:  # 交集为空，实际a不可选, 这里有个api的bug，tensorflow2.0会返回len，tensorflow2.0-gpu会返回0
            return -1
        return a

    def save_sqls(self):
        if self.time_log:
            for sql in self.sqls:
                self.sql_record.write(sql + '\n')
        self.sql_record.close()

    def get_value(self, s, time_step):
        value = self.critic.time_step_output(s, time_step)
        return value

    def call(self, inputs, training=None, mask=None):
        all_action = self.policy(inputs)
        return all_action

    def reset_policy_step(self):
        self.policy.time_step = 0

    def save_ac(self, path):
        self.policy.save_weights(os.path.join(path, self.model_name + '_REINFORCE.h5'))

    def load_ac(self, path):
        self.policy.load_weights(os.path.join(path, self.model_name + '_REINFORCE.h5'))

    def end_of_episode_actions(self, final_reward, ep_steps):
        self.episode += 1
        self.episode_rewards.append(final_reward)
        discounted_ep_rs = self.discount_reward(self.buffer.rewards, GAMMA, final_reward, ep_steps)
        # print("discounted_ep_rs:", discounted_ep_rs)
        episode = np.hstack((self.buffer.states, discounted_ep_rs, self.buffer.actions))
        index = (self.episode - 1) % BATCH_SIZE
        self.memory[index, :] = episode
        self.buffer.clear()

        if (not self.episode % self.aggregate_stats_window) or self.episode == self.max_episodes:
            # Find average, min, and max rewards over teh aggregate window
            reward_set = self.episode_rewards[-self.aggregate_stats_window:]
            average_reward = sum(reward_set) / len(reward_set)
            min_reward = min(reward_set)
            max_reward = max(reward_set)
            self.aggregate_episode_rewards['episode'].append(self.episode)
            self.aggregate_episode_rewards['avg_rewards'].append(average_reward)
            self.aggregate_episode_rewards['min_rewards'].append(min_reward)
            self.aggregate_episode_rewards['max_rewards'].append(max_reward)

        # show ongoing training stats
        if not self.episode % self.show_stats_interval:
            time_delta = datetime.now() - self.last_ep_start_time
            delta_min = (time_delta.seconds // 60)
            delta_sec = (time_delta.seconds - delta_min * 60) % 60
            print("Episode: {}, average reward: {}, time to complete: {} minutes, {} seconds".
                  format(self.episode, self.aggregate_episode_rewards['avg_rewards'][-1], delta_min, delta_sec))
            self.last_ep_start_time = datetime.now()

        if self.episode == self.max_episodes:
            time_delta = datetime.now() - self.start_time
            delta_hour = time_delta.seconds // 3600
            delta_min = ((time_delta.seconds - (delta_hour * 3600)) // 60)
            delta_sec = (time_delta.seconds - delta_hour * 3600 - delta_min * 60) % 60
            print('--------- TRAINING COMPLETE -----------')
            print("Episode: {}, average reward: {}, time to complete: {} minutes, {} seconds".
                  format(self.episode, self.aggregate_episode_rewards['avg_rewards'][-1], delta_min, delta_sec))

        # save models
        # if (not self.episode % self.save_models_interval) or (self.episode == self.max_episodes):
        #     self.save_ac(self.save_path)
        #     print('model saved')

    def pre_run(self):
        if self.pretrain_episode + BATCH_SIZE > self.pre_memory.shape[0]:
            self.pretrain_episode = 0
        pre_data = self.pre_memory[self.pretrain_episode: self.pretrain_episode + BATCH_SIZE]
        self.pretrain_episode += BATCH_SIZE
        print("pre train epoch:{}".format(self.pretrain_episode))
        self.pre_learn(pre_data)
        self.save_ac(self.save_path)

    def pre_learn(self, pre_data):
        with tf.GradientTape() as tape1:
            inputs = tf.constant(np.vstack(pre_data[:, :SEQ_LENGTH]), dtype=tf.float32)  # 只是堆叠数组
            # logits, values = self.call(inputs)
            logits = self.call(inputs)
            logits = tf.reshape(logits, [-1, self.state_size])  # [b * seq, total_words], dtype = tf.float32
            reward = tf.cast(tf.reshape(pre_data[:, SEQ_LENGTH: SEQ_LENGTH * 2], [-1]), dtype=tf.float32)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.cast(tf.reshape(pre_data[:, -SEQ_LENGTH:], [-1]), dtype=tf.int32),
                logits=logits)  # [b * seq]
            policy_loss = neg_log_prob * reward
            policy_loss_mean = tf.reduce_sum(policy_loss) / BATCH_SIZE
        policy_grades = tape1.gradient(policy_loss_mean, self.policy.trainable_weights)
        self.policy_optimizer.apply_gradients(zip(policy_grades, self.policy.trainable_weights))
        print("pre learning epoch done")

    def run_gen_sub(self):
        sub_count = 0
        t_count = 0
        tpath = os.path.abspath('.') + '/' + self.env.dbname + '/' + self.env.task_name
        nest_time = open(tpath + '_nest_time', 'w')
        nest_sql = open(tpath + '_nest_sql', 'w')
        last_record = -1
        try:
            nest_time.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), sub_count, t_count))
            while sub_count < 100:
                current_state = self.env.reset()
                self.reset_policy_step()  # time_step调到0
                reward, done = self.env.bug_reward, False
                ep_steps = 0
                nest_flag = False
                while not (done or ep_steps >= SEQ_LENGTH):
                    candidate_action = self.env.observe(current_state)
                    # self.check_action(candidate_action)
                    action = self.select_action(tf.constant(np.array([current_state])[None, :], dtype=tf.float32),
                                                candidate_action, time_step=ep_steps)
                    if self.env.c_obj.num_word_map[action] == 'subquery':
                        # print('gen sub')
                        nest_flag = True
                    # print('select action index:', action, ' ', self.env.num_word_map[action])
                    reward, done = self.env.step(action)
                    self.buffer.store(current_state, action, reward, ep_steps)  # 单步为0
                    ep_steps += 1
                    current_state = action
                if reward == self.env.bug_reward:
                    self.buffer.clear()  # 采样忽略
                    print('wrong grammar')
                    print(self.env.get_sql())
                else:
                    t_count += 1
                    if ep_steps == SEQ_LENGTH:
                        self.end_of_episode_actions(-2, ep_steps)
                    else:
                        self.end_of_episode_actions(reward, ep_steps)
                        sql = self.env.get_sql()
                        if self.env.is_satisfy(sql) and nest_flag is True:
                            sub_count += 1
                            self.sqls.append(sql)
                        else:
                            pass
                    if sub_count % 10 == 0 and sub_count != last_record:
                        print("time:{};s_count:{};t_count:{}\n".format(str(time.time()), sub_count, self.episode))
                        # time.sleep(1)
                        nest_time.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), sub_count, t_count))
                        last_record = sub_count
                    if self.episode % BATCH_SIZE == 0:  # 满了一个batch
                        # print("------learning Batch:-------------", self.episode // BATCH_SIZE)
                        self.learn()
            print("time:{};s_count:{};t_count:{}\n".format(str(time.time()), sub_count, self.episode))
            # time.sleep(1)
            nest_time.write("time:{};s_count:{};t_count:{}\n". format(str(time.time()), sub_count, t_count))
            for sql in self.sqls:
                nest_sql.write(sql + '\n')
            nest_sql.close()
            nest_time.close()
        except KeyboardInterrupt:
            print("time:{};s_count:{};t_count:{}\n".format(str(time.time()), sub_count, self.episode))
            # time.sleep(1)
            nest_time.write("time:{};s_count:{};t_count:{}\n".
                            format(str(time.time()), sub_count, t_count))
            for sql in self.sqls:
                nest_sql.write(sql + '\n')
            nest_sql.close()
            nest_time.close()

    def run_gen_insert(self):
        insert_count = 0
        t_count = 0
        tpath = os.path.abspath('.') + '/' + self.env.dbname + '/' + self.env.task_name
        insert_time = open(tpath + '_insert_time', 'w')
        insert_sql = open(tpath + '_insert_sql', 'w')
        last_record = -1
        try:
            insert_time.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), insert_count, t_count))
            while insert_count < 100:
                current_state = self.env.reset()
                self.reset_policy_step()  # time_step调到0
                reward, done = self.env.bug_reward, False
                ep_steps = 0
                insert_flag = False
                while not (done or ep_steps >= SEQ_LENGTH):
                    candidate_action = self.env.observe(current_state)
                    # self.check_action(candidate_action)
                    action = self.select_action(tf.constant(np.array([current_state])[None, :], dtype=tf.float32),
                                                candidate_action, time_step=ep_steps)
                    if self.env.c_obj.num_word_map[action] == 'insert':
                        # print('gen sub')
                        insert_flag = True
                    # print('select action index:', action, ' ', self.env.num_word_map[action])
                    reward, done = self.env.step(action)
                    self.buffer.store(current_state, action, reward, ep_steps)  # 单步为0
                    ep_steps += 1
                    current_state = action
                if reward == self.env.bug_reward:
                    self.buffer.clear()  # 采样忽略
                    print('wrong grammar')
                    print(self.env.get_sql())
                else:
                    t_count += 1
                    if ep_steps == SEQ_LENGTH:
                        self.end_of_episode_actions(-2, ep_steps)
                    else:
                        self.end_of_episode_actions(reward, ep_steps)
                        sql = self.env.get_sql()
                        if self.env.is_satisfy(sql) and insert_flag is True:
                            insert_count += 1
                            self.sqls.append(sql)
                        else:
                            pass
                    if insert_count % 10 == 0 and insert_count != last_record:
                        print("time:{};s_count:{};t_count:{}\n".format(str(time.time()), insert_count, self.episode))
                        # time.sleep(1)
                        insert_time.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), insert_count, t_count))
                        last_record = insert_count
                    if self.episode % BATCH_SIZE == 0:  # 满了一个batch
                        # print("------learning Batch:-------------", self.episode // BATCH_SIZE)
                        self.learn()
            print("time:{};s_count:{};t_count:{}\n".format(str(time.time()), insert_count, self.episode))
            # time.sleep(1)
            insert_time.write("time:{};s_count:{};t_count:{}\n". format(str(time.time()), insert_count, t_count))
            for sql in self.sqls:
                insert_sql.write(sql + '\n')
            insert_sql.close()
            insert_time.close()
        except KeyboardInterrupt:
            print("time:{};s_count:{};t_count:{}\n".format(str(time.time()), insert_count, self.episode))
            # time.sleep(1)
            insert_time.write("time:{};s_count:{};t_count:{}\n".
                            format(str(time.time()), insert_count, t_count))
            for sql in self.sqls:
                insert_sql.write(sql + '\n')
            insert_sql.close()
            insert_time.close()

    def run_gen_update(self):
        update_count = 0
        t_count = 0
        tpath = os.path.abspath('.') + '/' + self.env.dbname + '/' + self.env.task_name
        update_time = open(tpath + '_update_time', 'w')
        update_sql = open(tpath + '_update_sql', 'w')
        last_record = -1
        try:
            update_time.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), update_count, t_count))
            while update_count < 100:
                current_state = self.env.reset()
                self.reset_policy_step()  # time_step调到0
                reward, done = self.env.bug_reward, False
                ep_steps = 0
                update_flag = False
                while not (done or ep_steps >= SEQ_LENGTH):
                    candidate_action = self.env.observe(current_state)
                    # self.check_action(candidate_action)
                    action = self.select_action(tf.constant(np.array([current_state])[None, :], dtype=tf.float32),
                                                candidate_action, time_step=ep_steps)
                    if self.env.c_obj.num_word_map[action] == 'update':
                        # print('gen sub')
                        update_flag = True
                    # print('select action index:', action, ' ', self.env.num_word_map[action])
                    reward, done = self.env.step(action)
                    self.buffer.store(current_state, action, reward, ep_steps)  # 单步为0
                    ep_steps += 1
                    current_state = action
                if reward == self.env.bug_reward:
                    self.buffer.clear()  # 采样忽略
                    print('wrong grammar')
                    print(self.env.get_sql())
                else:
                    t_count += 1
                    if ep_steps == SEQ_LENGTH:
                        self.end_of_episode_actions(-2, ep_steps)
                    else:
                        self.end_of_episode_actions(reward, ep_steps)
                        sql = self.env.get_sql()
                        if self.env.is_satisfy(sql) and update_flag is True:
                            update_count += 1
                            self.sqls.append(sql)
                        else:
                            pass
                    if update_count % 10 == 0 and update_count != last_record:
                        print("time:{};s_count:{};t_count:{}\n".format(str(time.time()), update_count, self.episode))
                        # time.sleep(1)
                        update_time.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), update_count, t_count))
                        last_record = update_count
                    if self.episode % BATCH_SIZE == 0:  # 满了一个batch
                        # print("------learning Batch:-------------", self.episode // BATCH_SIZE)
                        self.learn()
            print("time:{};s_count:{};t_count:{}\n".format(str(time.time()), update_count, self.episode))
            # time.sleep(1)
            update_time.write("time:{};s_count:{};t_count:{}\n". format(str(time.time()), update_count, t_count))
            for sql in self.sqls:
                update_sql.write(sql + '\n')
            update_sql.close()
            update_time.close()
        except KeyboardInterrupt:
            print("time:{};s_count:{};t_count:{}\n".format(str(time.time()), update_count, self.episode))
            # time.sleep(1)
            update_time.write("time:{};s_count:{};t_count:{}\n".
                            format(str(time.time()), update_count, t_count))
            for sql in self.sqls:
                update_sql.write(sql + '\n')
            update_sql.close()
            update_time.close()

    def run_gen_delete(self):
        delete_count = 0
        t_count = 0
        tpath = os.path.abspath('.') + '/' + self.env.dbname + '/' + self.env.task_name
        delete_time = open(tpath + '_delete_time', 'w')
        delete_sql = open(tpath + '_delete_sql', 'w')
        last_record = -1
        try:
            delete_time.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), delete_count, t_count))
            while delete_count < 100:
                current_state = self.env.reset()
                self.reset_policy_step()  # time_step调到0
                reward, done = self.env.bug_reward, False
                ep_steps = 0
                delete_flag = False
                while not (done or ep_steps >= SEQ_LENGTH):
                    candidate_action = self.env.observe(current_state)
                    # self.check_action(candidate_action)
                    action = self.select_action(tf.constant(np.array([current_state])[None, :], dtype=tf.float32),
                                                candidate_action, time_step=ep_steps)
                    if self.env.c_obj.num_word_map[action] == 'delete':
                        # print('gen sub')
                        delete_flag = True
                    # print('select action index:', action, ' ', self.env.num_word_map[action])
                    reward, done = self.env.step(action)
                    self.buffer.store(current_state, action, reward, ep_steps)  # 单步为0
                    ep_steps += 1
                    current_state = action
                if reward == self.env.bug_reward:
                    self.buffer.clear()  # 采样忽略
                    print('wrong grammar')
                    print(self.env.get_sql())
                else:
                    t_count += 1
                    if ep_steps == SEQ_LENGTH:
                        self.end_of_episode_actions(-2, ep_steps)
                    else:
                        self.end_of_episode_actions(reward, ep_steps)
                        sql = self.env.get_sql()
                        if self.env.is_satisfy(sql) and delete_flag is True:
                            delete_count += 1
                            self.sqls.append(sql)
                        else:
                            pass
                    if delete_count % 10 == 0 and delete_count != last_record:
                        print("time:{};s_count:{};t_count:{}\n".format(str(time.time()), delete_count, self.episode))
                        # time.sleep(1)
                        delete_time.write(
                            "time:{};s_count:{};t_count:{}\n".format(str(time.time()), delete_count, t_count))
                        last_record = delete_count
                    if self.episode % BATCH_SIZE == 0:  # 满了一个batch
                        # print("------learning Batch:-------------", self.episode // BATCH_SIZE)
                        self.learn()
            print("time:{};s_count:{};t_count:{}\n".format(str(time.time()), delete_count, self.episode))
            # time.sleep(1)
            delete_time.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), delete_count, t_count))
            for sql in self.sqls:
                delete_sql.write(sql + '\n')
            delete_sql.close()
            delete_time.close()
        except KeyboardInterrupt:
            print("time:{};s_count:{};t_count:{}\n".format(str(time.time()), delete_count, self.episode))
            # time.sleep(1)
            delete_time.write("time:{};s_count:{};t_count:{}\n".
                              format(str(time.time()), delete_count, t_count))
            for sql in self.sqls:
                delete_sql.write(sql + '\n')
            delete_sql.close()
            delete_time.close()

    def run(self):
        s_count = 0
        bs_count = 0
        mx_ac = 0
        try:
            if self.time_log:
                self.efficiency.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), s_count, self.episode))
            while self.episode < self.max_episodes or (self.time_log and s_count < 1000):   # 前面算ac，后面算time
                current_state = self.env.reset()
                self.reset_policy_step()  # time_step调到0
                reward, done = self.env.bug_reward, False
                ep_steps = 0
                while not (done or ep_steps >= SEQ_LENGTH):
                    candidate_action = self.env.observe(current_state)
                    # self.check_action(candidate_action)
                    action = self.select_action(tf.constant(np.array([current_state])[None, :], dtype=tf.float32),
                                                candidate_action, time_step=ep_steps)
                    # print('select action index:', action, ' ', self.env.num_word_map[action])
                    reward, done = self.env.step(action)
                    self.buffer.store(current_state, action, reward, ep_steps)  # 单步为0
                    ep_steps += 1
                    current_state = action
                # if ep_steps == SEQ_LENGTH:
                #     self.end_of_episode_actions(reward, ep_steps)
                #     # self.buffer.clear()  # 采样忽略
                if reward == self.env.bug_reward:
                    self.buffer.clear()  # 采样忽略
                    print('wrong grammar')
                    print(self.env.get_sql())
                else:
                    if ep_steps == SEQ_LENGTH:
                        self.end_of_episode_actions(-2, ep_steps)
                    else:
                        self.end_of_episode_actions(reward, ep_steps)
                        sql = self.env.get_sql()
                        if self.env.is_satisfy(sql):
                            # print(sql)
                            s_count += 1
                            bs_count += 1
                            self.sqls.append(sql)
                        else:
                            pass
                    if self.time_log and s_count % 10 == 0:
                        print("time:{};s_count:{};t_count:{}\n".format(str(time.time()), s_count, self.episode))
                        # time.sleep(1)
                        self.efficiency.write("time:{};s_count:{};t_count:{}\n".
                                              format(str(time.time()), s_count, self.episode))
                    if self.episode % BATCH_SIZE == 0:  # 满了一个batch
                        # print("------learning Batch:-------------", self.episode // BATCH_SIZE)
                        if self.ac_log:
                            cur_ac = bs_count / BATCH_SIZE
                            bs_count = 0
                            print('cur_ac:', cur_ac)
                            if cur_ac > mx_ac:
                                print(cur_ac)
                                self.save_ac(self.save_path)
                                print('model saved')
                                mx_ac = cur_ac
                                self.accuracy.write(str(mx_ac) + '\n')
                            # self.curves.write(str(self.aggregate_episode_rewards))
                        self.learn()
            self.save_metric(s_count)
        except KeyboardInterrupt:
            self.save_metric(s_count)

    def save_metric(self, s_count):
        if self.time_log:
            self.efficiency.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), s_count, self.episode))
            self.efficiency.close()
            self.save_sqls()
        if self.ac_log:
            self.accuracy.close()
            self.curves.write(str(self.aggregate_episode_rewards))
            self.curves.close()

    def learn(self):
        with tf.GradientTape() as tape1:
            inputs = tf.constant(np.vstack(self.memory[:, :SEQ_LENGTH]), dtype=tf.float32)   # 只是堆叠数组
            logits = self.call(inputs)
            logits = tf.reshape(logits, [-1, self.state_size])   # [b * seq, total_words], dtype = tf.float32
            reward = tf.cast(tf.reshape(self.memory[:, SEQ_LENGTH: SEQ_LENGTH * 2], [-1]), dtype=tf.float32)
            # policy = tf.nn.softmax(logits, axis=1)  # [b * seq, total_words]
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits\
                (labels=tf.cast(tf.reshape(self.memory[:, -SEQ_LENGTH:], [-1]), dtype=tf.int32), logits=logits)  # [b * seq]
            policy_loss = neg_log_prob * reward
            # entropy bonus
            policy = tf.nn.softmax(logits, axis=1)
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits)
            policy_loss = policy_loss - ENTROPY_BETA * entropy
            policy_loss_mean = tf.reduce_sum(policy_loss) / BATCH_SIZE
        policy_grades = tape1.gradient(policy_loss_mean, self.policy.trainable_weights)
        self.policy_optimizer.apply_gradients(zip(policy_grades, self.policy.trainable_weights))

    def discount_reward(self, r, gamma, final_r, ep_steps):
        # gamma 越大约有远见
        discounted_r = np.zeros(SEQ_LENGTH)
        discounted_r[ep_steps:] = final_r
        running_add = 0     # final_r已经存了
        for t in reversed(range(0, ep_steps)):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def plot_training_results(self):
        fig = plt.figure(figsize=(8, 8))
        fig.tight_layout(pad=0.5)
        plt.style.use('ggplot')

        plt.title('Policy Training Progress')
        plt.plot(np.arange(1, len(self.episode_rewards)+1), self.episode_rewards, label='eposide rewards')
        plt.plot(self.aggregate_episode_rewards['episode'], self.aggregate_episode_rewards['avg_rewards'], label="average rewards")
        # plt.plot(self.aggregate_episode_rewards['episode'], self.aggregate_episode_rewards['max_rewards'], label="max rewards")
        # plt.plot(self.aggregate_episode_rewards['episode'], self.aggregate_episode_rewards['min_rewards'], label="min rewards")
        plt.legend(loc=2)
        plt.grid(True)
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.show()

        if self.save_training_plot:
            plot_name = "training_progress_{}.png".format(self.model_name)
            fig.savefig(os.path.join(self.save_path, plot_name))

    # def inference(self, nums):
    #     print("begin inference ------------------------- ")
    #     count = 0
    #     s_count = 0
    #     while count < nums:
    #         current_state = self.env.reset()
    #         self.reset_policy_step()  # time_step调到0
    #         reward, done = self.env.bug_reward, False
    #         ep_steps = 0
    #         while not (done or ep_steps >= SEQ_LENGTH):
    #             candidate_action = self.env.observe(current_state)
    #             # self.check_action(candidate_action)
    #             action = self.select_action(tf.constant(np.array([current_state])[None, :], dtype=tf.float32),
    #                                         candidate_action, time_step=ep_steps)
    #
    #             # print('select action index:', action, ' ', self.env.num_word_map[action])
    #             reward, done = self.env.step(action)
    #             self.buffer.store(current_state, action, reward, ep_steps)  # 单步为0
    #             ep_steps += 1
    #             current_state = action
    #         if ep_steps == SEQ_LENGTH or reward == self.env.bug_reward:
    #             self.buffer.clear()  # 采样忽略
    #             # print('采样忽略')
    #
    #         else:
    #             sql = self.env.get_sql()
    #             count += 1
    #             if self.env.is_satisfy(sql):
    #                 print(sql)
    #                 s_count += 1
    #     ac = s_count / count
    #     # self.a_log.write(str(ac))
    #     self.a_log.close()


#%%
if __name__ == '__main__':
    para = sys.argv
    dbname = para[1]
    ctype = para[2]     # cost/card
    mtype = para[3]     # point/range(0/1)
    if mtype == 'point':
        target_type = 0
        metric = int(para[4])
    else:
        assert(mtype == 'range')
        target_type = 1
        metric = (int(para[4]), int(para[5]))
    if ctype == 'cost':
        task = envCost(metric=metric, dbname=dbname, target_type=target_type)
    else:
        assert(ctype == 'card')
        task = envCard(metric=metric, dbname=dbname, target_type=target_type)

    master = REINFORCE(env=task)
    for _ in range(100):
        master.pre_run()
    master.run()
    master.run_gen_update()
    # %%
    # master.inference(1000)
    master.plot_training_results()

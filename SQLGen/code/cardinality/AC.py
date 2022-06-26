import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from matplotlib import pyplot as plt
from cardinality.simple_env import *
import multiprocessing
import os
import time
from datetime import datetime
from pathlib import Path
import base
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')


########
workers_num = multiprocessing.cpu_count()
# workers_num = 2
EMB_DIM = 200   # embedding dimension
HIDDEN_DIM = 128    # hidden state dimension of lstm cell
SEQ_LENGTH = 20     # sequence length
SEED = 88
BATCH_SIZE = 64
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
class AC_Agent(keras.Model):
    def __init__(self, env):
        super(AC_Agent, self).__init__()
        self.env = env
        self.state_size = env.observation_space
        self.action_size = env.observation_space
        self.actor = Target_Lstm(self.state_size, self.action_size)
        # self.critic = Target_Lstm(self.state_size, 1)
        self.actor_optimizer = optimizers.Adam(0.001)
        # self.critic_optimizer = optimizers.Adam(0.003)

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
        # self.time_log =

        self.episode = 0
        self.max_episodes = BATCH_SIZE * 500

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

        self.ac_log = True
        self.time_log = False
        cpath = os.path.abspath('.')
        tpath = cpath + '/' + self.env.dbname + '/' + self.env.task_name
        if self.ac_log:
            self.ac_path = tpath + '_ac'
            self.a_log = open(self.ac_path, 'w')
        if self.time_log:
            print('cal_time...')
            time.sleep(5)
            self.sqls = set()
            sql_path = tpath + '_sql'
            t_path = tpath + '_time'
            self.t_log = open(t_path, 'w')
            self.sql_log = open(sql_path, 'w')

    def load_predata(self):
        cpath = os.path.abspath('.')
        tpath = cpath + '/' + self.env.dbname + '/' + self.env.task_name + '_predata.npy'
        pre_memory = np.load(tpath)
        return pre_memory

    def select_action(self, s, candidate_action, time_step):
        o_t = self.actor.time_step_output(s, time_step)[0]  # [b, total_words]
        mask_o_t = tf.where(candidate_action, o_t, 0)     # acts as a mask
        # self.check_action(candidate_prob)
        mask_o_t = tf.constant(mask_o_t)[None, :]
        # a = tf.random.categorical(tf.math.log(o_t), 1)[0]
        a = tf.random.categorical(mask_o_t, 1)[0]
        a = int(a)
        # 这里取交集可能为空, 在于初始化参数有的word本身概率为0
        if a == len(candidate_action) or candidate_action[a] == 0:  # 交集为空，实际a不可选, 这里有个api的bug，tensorflow2.0会返回len，tensorflow2.0-gpu会返回0
            a = self.random_select_action(candidate_action)
        # print("choose action: ", a, " ", gen_sql.num_word_map[a])
        return a

    def random_select_action(self, candidate_action):
        candidate_list = np.argwhere(candidate_action == np.max(candidate_action)).flatten()
        action = np.random.choice(candidate_list)
        return action

    def inference_select_action(self, s, candidate_action, time_step):
        o_t = self.actor.time_step_output(s, time_step)[0]  # [b, total_words]
        mask_o_t = tf.where(candidate_action, o_t, 0)  # acts as a mask
        # self.check_action(candidate_prob)
        # top_5 = tf.math.top_k(candidate_prob, 5)
        # top_index = top_5.indices.numpy()
        # condition = np.zeros(len(o_t), dtype=bool)
        # condition[top_index] = True
        # condition = tf.convert_to_tensor(condition)
        # candidate_prob = tf.where(condition, candidate_prob, 0)
        mask_o_t_c = tf.constant(mask_o_t)[None, :]

        a = tf.random.categorical(mask_o_t_c, 1)[0]
        a = int(a)
        print("choose:{}, prob:{}".format(a, mask_o_t[a]))
        if a == len(candidate_action) or candidate_action[a] == 0:  # 交集为空，实际a不可选, 这里有个api的bug，tensorflow2.0会返回len，tensorflow2.0-gpu会返回0
            return -1
        return a

    def get_value(self, s, time_step):
        value = self.critic.time_step_output(s, time_step)
        return value

    def call(self, inputs, training=None, mask=None):
        all_action = self.actor(inputs)
        # all_values = self.critic(inputs)
        # return all_action, all_values   # [batchsz, seq_len, total_words], [batchsz, seq_len, 1]
        return all_action

    def reset_actor_step(self):
        self.actor.time_step = 0

    def save_ac(self, path):
        self.actor.save_weights(os.path.join(path, self.model_name + '_actor.h5'))
        # self.critic.save_weights(os.path.join(path, self.model_name + '_critic.h5'))

    def load_ac(self, path):
        self.actor.load_weights(os.path.join(path, self.model_name + '_actor.h5'))
        # self.critic.load_weights(os.path.join(path, self.model_name + '_critic.h5'))
        # self.server(tf.constant(np.random.randint(size=SEQ_LENGTH, low=self.vob_dim)[None, :]))

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
            print("Episode: {}, average reward: {}, time to complete: {} minutes, {} seconds".format(self.episode, self.aggregate_episode_rewards['avg_rewards'][-1], delta_min, delta_sec))
            self.last_ep_start_time = datetime.now()

        if self.episode == self.max_episodes:
            time_delta = datetime.now() - self.start_time
            delta_hour = time_delta.seconds // 3600
            delta_min = ((time_delta.seconds - (delta_hour * 3600)) // 60)
            delta_sec = (time_delta.seconds - delta_hour * 3600 - delta_min * 60) % 60
            print('--------- TRAINING COMPLETE -----------')
            print("Episode: {}, average reward: {}, time to complete: {} minutes, {} seconds".format(self.episode,
                                                                                                     self.aggregate_episode_rewards[
                                                                                                         'avg_rewards'][
                                                                                                         -1], delta_min,
                                                                                                     delta_sec))

        # save models
        if (not self.episode % self.save_models_interval) or (self.episode == self.max_episodes):
            self.save_ac(self.save_path)
            print('model saved')

    def end_of_pretrain_episode_actions(self, final_reward, ep_steps):
        self.pretrain_episode += 1
        discounted_ep_rs = self.discount_reward(self.buffer.rewards, GAMMA, final_reward, ep_steps)
        # print("discounted_ep_rs:", discounted_ep_rs)
        episode = np.hstack((self.buffer.states, discounted_ep_rs, self.buffer.actions))
        index = (self.pretrain_episode - 1) % BATCH_SIZE
        self.memory[index, :] = episode
        self.buffer.clear()

        if not self.pretrain_episode % self.show_stats_interval:
            time_delta = datetime.now() - self.last_ep_start_time
            delta_min = (time_delta.seconds // 60)
            delta_sec = (time_delta.seconds - delta_min * 60) % 60
            print("Episode: {}, time to complete: {} minutes, {} seconds".format(self.pretrain_episode, delta_min, delta_sec))
            self.last_ep_start_time = datetime.now()

        if self.pretrain_episode == self.max_pretrain_episodes:
            time_delta = datetime.now() - self.start_time
            delta_hour = time_delta.seconds // 3600
            delta_min = ((time_delta.seconds - (delta_hour * 3600)) // 60)
            delta_sec = (time_delta.seconds - delta_hour * 3600 - delta_min * 60) % 60
            print('--------- PRE TRAINING COMPLETE -----------')
            print("Episode: {}, time to complete: {} minutes, {} seconds".format(self.pretrain_episode, delta_min, delta_sec))
            # self.start_time = datetime.now()

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
        actor_grades = tape1.gradient(policy_loss_mean, self.actor.trainable_weights)
        self.actor_optimizer.apply_gradients(zip(actor_grades, self.actor.trainable_weights))
        print("pre learning epoch done")

    def run(self):
        s_count = 0
        try:
            if self.time_log:
                self.t_log.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), s_count, self.episode))
            while self.episode < self.max_episodes or (self.time_log and s_count < 1000):   # 前面算ac，后面算time
                current_state = self.env.reset()
                self.reset_actor_step()  # time_step调到0
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
                if ep_steps == SEQ_LENGTH or reward == self.env.bug_reward:
                    self.buffer.clear()  # 采样忽略
                    # print('采样忽略')
                else:
                    self.end_of_episode_actions(reward, ep_steps)
                    if self.time_log:
                        if self.env.is_satisfy():
                            sql = self.env.get_sql()
                            # print(sql)
                            self.sqls.add(sql)
                            if len(self.sqls) > s_count:
                                s_count = len(self.sqls)
                                if s_count % 10 == 0:
                                    # print("time:{};s_count:{};t_count:{}\n".format(str(time.time()), s_count, self.episode))
                                    # time.sleep(1)
                                    self.t_log.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), s_count, self.episode))
                    if self.episode % BATCH_SIZE == 0:  # 满了一个batch
                        # print("---------------------learning Batch:------------------------", self.episode // BATCH_SIZE)
                        with open(self.ac_path, 'w') as f:
                            f.write(str(self.aggregate_episode_rewards))
                        self.learn()
            if self.time_log:
                for sql in self.sqls:
                    self.sql_log.write(sql + '\n')
                self.sql_log.close()
                self.t_log.close()
        except KeyboardInterrupt as result:
            if self.time_log:
                for sql in self.sqls:
                    self.sql_log.write(sql + '\n')
                self.t_log.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), s_count, self.episode))
                self.sql_log.close()
                self.t_log.close()
            if self.ac_log:
                with open(self.ac_path, 'w') as f:
                    f.write(str(self.aggregate_episode_rewards))


    def learn(self):
        with tf.GradientTape() as tape1:
            inputs = tf.constant(np.vstack(self.memory[:, :SEQ_LENGTH]), dtype=tf.float32)   # 只是堆叠数组
            logits = self.call(inputs)
            logits = tf.reshape(logits, [-1, self.state_size])   # [b * seq, total_words], dtype = tf.float32
            reward = tf.cast(tf.reshape(self.memory[:, SEQ_LENGTH: SEQ_LENGTH * 2], [-1]), dtype=tf.float32)
            # policy = tf.nn.softmax(logits, axis=1)  # [b * seq, total_words]
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(tf.reshape(self.memory[:, -SEQ_LENGTH:], [-1]), dtype=tf.int32), logits=logits)  # [b * seq]
            policy_loss = neg_log_prob * reward
            # entropy bonus
            policy = tf.nn.softmax(logits, axis=1)
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits)
            policy_loss = policy_loss - ENTROPY_BETA * entropy
            policy_loss_mean = tf.reduce_sum(policy_loss) / BATCH_SIZE
        actor_grades = tape1.gradient(policy_loss_mean, self.actor.trainable_weights)
        self.actor_optimizer.apply_gradients(zip(actor_grades, self.actor.trainable_weights))

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

        plt.title('Actor Critic Training Progress')
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

    def inference(self, nums):
        print("begin inference ------------------------- ")
        count = 0
        s_count = 0
        while count < nums:
            current_state = self.env.reset()
            self.reset_actor_step()  # time_step调到0
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
            if ep_steps == SEQ_LENGTH or reward == self.env.bug_reward:
                self.buffer.clear()  # 采样忽略
                # print('采样忽略')
            else:
                # sql = self.env.get_sql()
                count += 1
                if self.env.is_satisfy():
                    print(self.env.get_sql())
                    s_count += 1
        ac = s_count / count
        # self.a_log.write(str(ac))
        self.a_log.close()


#%%
if __name__ == '__main__':
    para = sys.argv
    dbname = para[1]
    mtype = para[2]   # point/range(0/1)
    if mtype == 'point':
        pc = int(para[3])
        task = GenSqlEnv(metric=pc, dbname=dbname, target_type=0)
    else:
        rc = (int(para[3]), int(para[4]))
        task = GenSqlEnv(metric=rc, dbname=dbname, target_type=1)
    master = AC_Agent(env=task)
    for _ in range(100):
        master.pre_run()
    master.run()
    # %%
    master.inference(1000)
    master.plot_training_results()

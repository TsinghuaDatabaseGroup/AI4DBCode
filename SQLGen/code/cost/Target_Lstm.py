import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import threading
from queue import Queue
from matplotlib import pyplot as plt
import time
from cost.env import *
import multiprocessing
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2"
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')


########
workers_num = multiprocessing.cpu_count()
# workers_num = 2
EMB_DIM = 200   # embedding dimension
HIDDEN_DIM = 128    # hidden state dimension of lstm cell
SEQ_LENGTH = 20     # sequence length
SEED = 88
BATCH_SIZE = 64
GAMMA = 0.98
ENTROPY_BETA = 0.01
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

        self.rnn_cell0 = layers.LSTMCell(HIDDEN_DIM, dropout=0.5)
        self.rnn_cell1 = layers.LSTMCell(HIDDEN_DIM, dropout=0.5)
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
class ACNet(keras.Model):
    def __init__(self, state_size, action_size):
        super(ACNet, self).__init__()
        self.actor = Target_Lstm(state_size, action_size)
        self.critic = Target_Lstm(state_size, 1)

    def select_action(self, s, candidate_action, time_step):
        o_t = self.actor.time_step_output(s, time_step)[0]  # [b, total_words]
        candidate_prob = tf.where(candidate_action, o_t, 0)
        # self.check_action(candidate_prob)
        o_t = tf.constant(candidate_prob)[None, :]
        a = tf.random.categorical(tf.math.log(o_t), 1)[0]
        a = int(a)
        # 这里取交集可能为空, 在于初始化参数有的word本身概率为0
        if candidate_action[a] == 0:  # 交集为空，实际a不可选
            a = tf.random.categorical(tf.math.log(tf.convert_to_tensor(candidate_action, dtype=tf.float32)[None, :]), 1)[0]
            a = int(a)
        # print("choose action: ", a, " ", gen_sql.num_word_map[a])
        return a

    def check_action(self, check_list):
        for i in range(len(check_list)):
            if check_list[i] != 0:
                print(gen_sql.num_word_map[i])

    def get_value(self, s, time_step):
        value = self.critic.time_step_output(s, time_step)
        return value

    def call(self, inputs, training=None, mask=None):
        all_action = self.actor(inputs)
        all_values = self.critic(inputs)
        return all_action, all_values   # [batchsz, seq_len, total_words], [batchsz, seq_len, 1]

    def reset_actor_step(self):
        self.actor.time_step = 0


#%%
class Worker(threading.Thread):
    def __init__(self, server, opt, result_queue, idx, task, vob_dim, sample_num):
        super(Worker, self).__init__()
        self.result_queue = result_queue
        self.server = server
        self.opt = opt
        self.client = ACNet(vob_dim, vob_dim)

        self.worker_idx = idx

        self.env = task
        # self.env.relation_tree.show()

        self.vob_dim = vob_dim
        self.memory = np.zeros((BATCH_SIZE, SEQ_LENGTH * 3))
        self.sample_num = sample_num
        self.workers_log = open(os.path.join('log/workers{}.log'.format(idx)), 'w')  # 主要是记录worker采样

    # def check_action(self, check_list):
    #     print('candidate check_list')
    #     for i in range(len(check_list)):
    #         if check_list[i] == 1:
    #             print(self.env.num_word_map[i])

    def run(self):
        buffer = Buffer()
        epi_counter = 0
        batch_rewards = 0.
        while epi_counter < self.sample_num:
            if epi_counter % 1000 == 0:
                print('worker{worker_id}: {epi_counter} sampled done'.format(worker_id=self.worker_idx, epi_counter=epi_counter))
                self.workers_log.write('worker{worker_id}: {epi_counter} sampled done'.format(worker_id=self.worker_idx, epi_counter=epi_counter))
            current_state = self.env.reset()
            self.client.reset_actor_step()  # time_step调到0
            done = False
            ep_steps = 0
            while not done:
                candidate_action = self.env.get_one_step_reward(current_state)
                # self.check_action(candidate_action)
                action = self.client.select_action(tf.constant(np.array([current_state])[None, :], dtype=tf.float32),
                                                   candidate_action, time_step=ep_steps)
                # print('select action index:', action, ' ', self.env.num_word_map[action])
                reward, done = self.env.step(action)
                new_state = action

                if ep_steps >= SEQ_LENGTH - 1 or done:
                    if not done:
                        buffer.clear()
                        # print("restart....")
                        break   # 直接忽略此采样, 如何改进呢？
                    else:
                        # 最后一次传的是一个tuple (terminal_action,  episode_reward)
                        success = reward[0]
                        if success:
                            buffer.store(current_state, action, self.env.step_reward, ep_steps)  # 单步为0
                            batch_rewards += reward[1]
                            discounted_ep_rs = self.discount_reward(buffer.rewards, GAMMA, reward[1], ep_steps)
                            # print("discounted_ep_rs:", discounted_ep_rs)
                            episode = np.hstack((buffer.states, discounted_ep_rs, buffer.actions))
                            index = epi_counter % BATCH_SIZE
                            epi_counter += 1  # 有效sample
                            self.memory[index, :] = episode
                            buffer.clear()
                            if index == BATCH_SIZE - 1:   # 满了一个batch
                                with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                                    policy_loss,  value_loss = self.compute_loss(self.memory[:, :SEQ_LENGTH],
                                                                                 self.memory[:, SEQ_LENGTH: SEQ_LENGTH * 2],
                                                                                 self.memory[:, -SEQ_LENGTH:])
                                actor_grades = tape1.gradient(policy_loss, self.client.actor.trainable_weights)
                                critic_grades = tape2.gradient(value_loss, self.client.critic.trainable_weights)
                                self.opt['actor_optimizer'].apply_gradients(zip(actor_grades, self.server.actor.trainable_weights))
                                self.opt['critic_optimizer'].apply_gradients(zip(critic_grades, self.server.critic.trainable_weights))
                                self.client.set_weights(self.server.get_weights())

                                self.result_queue.put((policy_loss, value_loss, batch_rewards / BATCH_SIZE))
                                print(self.worker_idx, batch_rewards / BATCH_SIZE)
                                break
                        else:
                            # 数据库返回有问题，这次采样失败
                            print('database refused, sample failed')
                            buffer.clear()
                else:
                    buffer.store(current_state, action, reward, ep_steps)
                    ep_steps += 1
                    current_state = new_state
        # self.result_queue.put(None)

    @tf.function
    def compute_loss(self, states, rewards, action):
        inputs = tf.constant(np.vstack(states), dtype=tf.float32)   # 只是堆叠数组

        logits, values = self.client(inputs)
        logits = tf.reshape(logits, [-1, self.vob_dim])   # [b * seq, total_words], dtype = tf.float32

        values = tf.reshape(values, [-1])    # [b * seq1]   # tf.float32
        # 计算advantage = R() - v(s)
        # rewards [b, seq]
        advantage = tf.cast(tf.reshape(rewards, [-1]), dtype=tf.float32) - values
        # critic的loss
        value_loss = tf.square(advantage)    # [b * seq1]
        # 策略损失
        policy = tf.nn.softmax(logits, axis=1)  # [b * seq, total_words]

        neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(tf.reshape(action, [-1]), dtype=tf.int32), logits=logits)  # [b * seq]

        policy_loss = neg_log_prob * tf.stop_gradient(advantage)
        # entropy bonus
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits)
        policy_loss = policy_loss - ENTROPY_BETA * entropy
        return tf.reduce_sum(policy_loss) / BATCH_SIZE, tf.reduce_mean(value_loss)

    def discount_reward(self, r, gamma, final_r, ep_steps):
        # gamma越大约有远见
        discounted_r = np.zeros(SEQ_LENGTH)
        discounted_r[ep_steps:] = final_r
        running_add = final_r
        for t in reversed(range(0, ep_steps)):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r


class Agent:
    # 总控
    def __init__(self, vob_dim):
        self.opt = {
            'actor_optimizer': optimizers.Adam(1e-3),
            'critic_optimizer': optimizers.Adam(3e-3),
        }
        self.agent_log = open(os.path.join('log/agent.log'), 'w')  # 主要是记录agent状态
        self.vob_dim = vob_dim
        self.server = ACNet(vob_dim, vob_dim)
        self.load_weights()

    def train(self, sample_num):
        res_queue = Queue()
        workers = [
            Worker(self.server, self.opt, res_queue, i, GenSqlEnv(metric=300000),
                   self.vob_dim, sample_num) for i in range(workers_num)]
        for i, worker in enumerate(workers):
            print('starting worker {}'.format(i))
            self.agent_log.write('starting worker {}'.format(i))
            worker.start()
            # returns
        [w.join() for w in workers]

        self.save_weights(sample_num)

        policy_loss = []
        value_loss = []
        rewards = []
        for _ in range(res_queue.qsize()):
            returns = res_queue.get()
            policy_loss.append(returns[0])
            value_loss.append(returns[1])
            rewards.append(returns[2])

        print('total batches:', len(policy_loss))

        self.save_svg(policy_loss, name='policy_loss')
        self.save_svg(value_loss, name='value_loss')
        self.save_svg(rewards, name='rewards')

    def save_svg(self, save_list, name):
        plt.figure()
        plt.plot(np.arange(len(save_list)), save_list)
        plt.xlabel('batch')
        plt.ylabel(name)
        plt.savefig(name + '.svg')

    def test(self, sample_num):
        epi_counter = 0
        average_cost = 0.
        average_rewards = 0.
        while epi_counter < sample_num:
            cur_state = gen_sql.reset()
            self.server.reset_actor_step()  # time_step调到0
            done = False
            ep_steps = 0
            while not done:
                candidate_action = gen_sql.get_one_step_reward(cur_state)
                action = self.server.select_action(tf.constant(np.array([cur_state])[None, :], dtype=tf.float32),
                                                   candidate_action, time_step=ep_steps)
                # print('select action index:', action, ' ', self.env.num_word_map[action])
                reward, done = gen_sql.step(action)
                new_state = action

                if ep_steps >= SEQ_LENGTH - 1 or done:
                    if not done:
                        break  # 直接忽略此采样, 如何改进呢？
                    else:
                        success = reward[0]
                        if success:
                            average_rewards += reward[1]
                            print(reward[1])
                            epi_counter += 1
                        else:
                            # 数据库返回有问题，这次采样失败
                            print('database refused, sample failed')

                else:
                    ep_steps += 1
                    cur_state = new_state
        print(average_rewards / sample_num)

    def save_weights(self, sample_num):
        self.server.save_weights('./server_weights')
        self.agent_log.write("Time:{time} saved. episodes num:{total_sample_num}".format(
            time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            total_sample_num=sample_num * workers_num))

    def load_weights(self):
        if os.path.exists('./server_weights'):
            self.server.load_weights('./server_weights')
            print('server load weights success')
        else:
            self.server(tf.constant(np.random.randint(size=SEQ_LENGTH, low=self.vob_dim)[None, :]))



#if __name__ == '__main__':

#%%



#%%
gen_sql = GenSqlEnv(metric=300000)
master = Agent(vob_dim=gen_sql.observation_space)

print(master.test(1000))
print(master.test(1000))


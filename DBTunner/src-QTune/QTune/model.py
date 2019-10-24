import sys
import datetime
from os import path
import subprocess
import time
from collections import deque
import numpy as np
import random
import tensorflow as tf
import pandas

import pymysql
import pymysql.cursors as pycursor

import gym
from gym import spaces
from gym.utils import seeding

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler


def target_range(x, target_min=a_low, target_max=a_high):
    x02 = K.tanh(x) + 1  # x in range(0,2)
    scale = (target_max - target_min) / 2.
    return x02 * scale + target_min



# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value
# Tunable parameters
# learning_rate = 0.001
# epsilon = 1.0
# epsilon_decay = .995
# gamma = .95
# tau   = .125
# 4*relu
class ActorCritic:
    def __init__(self, env, sess, learning_rate=0.001, train_min_size=32, size_mem=2000, size_predict_mem=2000):
        self.env = env

        self.sess = sess

        self.learning_rate = learning_rate  # 0.001
        self.train_min_size = train_min_size
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = .095
        self.tau = .125

        # ===================================================================== #
        #                               Actor Model                             #
        # Chain rule: find the gradient of chaging the actor network params in  #
        # getting closest to the final value network predictions, i.e. de/dA    #
        # Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
        # ===================================================================== #
        self.memory = deque(maxlen=size_mem)
        self.mem_predicted = deque(maxlen=size_predict_mem)
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32,
                                                [None, self.env.action_space.shape[
                                                    0]])  # where we will feed de/dC (from critic)

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output,
                                        actor_model_weights, -self.actor_critic_grad)  # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #

        self.critic_state_input, self.critic_action_input, \
        self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output,
                                         self.critic_action_input)  # where we calcaulte de/dC for feeding above

        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    # ========================================================================= #
    #                              Model Definitions                            #
    # ========================================================================= #

    def create_actor_model(self):
        state_input = Input(shape=self.env.observation_space.shape)

        h1 = Dense(128, activation='relu')(state_input)
        # n1 = BatchNormalization()(h1)
        h2 = Dense(64, activation='relu')(h1)
        d1 = Dropout(0.3)(h2)
        # add a dense-tanh expend the space!!
        output = Dense(self.env.action_space.shape[0], activation=target_range)(d1)

        model = Model(input=state_input, output=output)
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        # (dense dense)->dense->dense->BN->dense
        state_input = Input(shape=self.env.observation_space.shape)
        state_h1 = Dense(128)(state_input)
        # state_h2 = Dense(13)(state_h1)

        action_input = Input(shape=self.env.action_space.shape)
        action_h1 = Dense(128)(action_input)  #

        merged = Add()([state_h1, action_h1])
        merged_h1 = Dense(256, activation='relu')(merged)
        h2 = Dense(256)(merged_h1)
        h3 = Dense(64, activation='tanh')(h2)
        d1 = Dropout(0.3)(h3)
        n1 = BatchNormalization()(d1)
        output = Dense(1)(n1)

        model = Model(input=[state_input, action_input], output=output)

        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])
        # print("Mem: Q-%f"%reward)

    def _train_actor(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, _ = sample
            predicted_action = self.actor_model.predict(cur_state)

            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input: cur_state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: cur_state,
                self.actor_critic_grad: grads
            })

    def _train_critic(self, samples):
        for sample in samples:
            cur_state, action, t_reward, new_state, done = sample
            reward = np.array([])
            reward = np.append(reward, t_reward[0])

            # print("<>Q-value:")
            # print(reward)
            # if not done:
            target_action = self.target_actor_model.predict(new_state)
            future_reward = self.target_critic_model.predict(
                [new_state, target_action])[0][0]
            reward += self.gamma * future_reward
            # There comes the convert
            # print("Look:")
            # print(cur_state.shape)
            # print(action.shape)
            # print(reward.shape)
            # print(reward)
            self.critic_model.fit([cur_state, action], reward, verbose=0)  # update the Q-value

    def train(self):
        self.batch_size = self.train_min_size  # 32
        if len(self.memory) < self.batch_size:
            return

        rewards = []
        samples = random.sample(self.memory, self.batch_size - 2)
        # print(samples)
        self._train_critic(samples)
        self._train_actor(samples)
        self.update_target()

    # ========================================================================= #
    #                         Target Model Updating                             #
    # ========================================================================= #

    def _update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_actor_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.target_critic_model.set_weights(critic_target_weights)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def act(self, cur_state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon or len(self.memory) < self.batch_size:
            print("<>Random Choose")
            return self.env.action_space.sample(), 0

        print("<>Predicted Action")
        action = self.actor_model.predict(cur_state)
        for i in range(action[0].shape[0]):
            if action[0][i] <= self.env.default_action[i]:
                print("<>Action %d Predict Fail: %f" % (i, action[0][i]))
                action[0][i] = self.env.default_action[i]
                # return self.env.action_space.sample()
            elif action[0][i] > self.env.a_high[i]:
                print("<>Action %d Predict Highest: %f" % (i, action[0][i]))
                action[0][i] = self.env.a_high[i]

        return action[0], 1

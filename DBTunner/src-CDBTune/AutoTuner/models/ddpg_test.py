# -*- coding: utf-8 -*-
"""
Deep Deterministic Policy Gradient Model Test

"""

import gym
import numpy as np
from ddpg import DDPG
from itertools import count

config = {
    'model': '',
    'alr': 0.001,
    'clr': 0.001,
    'gamma': 0.9,
    'batch_size': 32,
    'tau': 0.002
}

env = gym.make('MountainCarContinuous-v0')  # ('Hopper-v1')
print(env.action_space, env.observation_space)
print(env.action_space.low, env.action_space.high)
n_actions = 1
n_states = 2

ddpg = DDPG(
    n_actions=n_actions,
    n_states=n_states,
    opt=config
)

returns = []
for i in xrange(10000):
    ddpg.reset(0.1)
    state = env.reset()
    total_reward = 0.0
    for t in count():

        action = ddpg.choose_action(state)
        next_state, reward, done = ddpg.apply_action(env, action)
        # env.render()
        ddpg.replay_memory.push(
            state=state,
            action=action,
            next_state=next_state,
            terminate=done,
            reward=reward
        )
        total_reward += reward

        state = next_state
        if done:
            break

        if len(ddpg.replay_memory) > 100:
            ddpg.update()
    returns.append(total_reward)
    print("Episode: {} Return: {} Mean Return: {} STD: {}".format(i, total_reward, np.mean(returns), np.std(returns)))


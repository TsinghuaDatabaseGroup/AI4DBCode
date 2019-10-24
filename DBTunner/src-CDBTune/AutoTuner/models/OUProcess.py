# -*- coding: utf-8 -*-

"""
Ornstein–Uhlenbeck process
"""

import numpy as np


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUProcess(object):

    def __init__(self, n_actions, theta=0.15, mu=0, sigma=0.1, ):

        self.n_actions = n_actions
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.current_value = np.ones(self.n_actions) * self.mu

    def reset(self, sigma=0):
        self.current_value = np.ones(self.n_actions) * self.mu
        if sigma != 0:
            self.sigma = sigma

    def noise(self):
        x = self.current_value
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.current_value = x + dx
        return self.current_value


if __name__ == '__main__':
    ou = OUProcess(3, theta=0.3)
    states = []
    for i in range(1000):
        states.append(ou.noise())
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()

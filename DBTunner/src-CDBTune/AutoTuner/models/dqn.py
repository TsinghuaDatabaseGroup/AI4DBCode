# -*- coding: utf-8 -*-
"""
Deep Q Network
"""


class DQN(object):

    def __init__(self):
        pass

    def update(self):
        """ Update the Actor and Critic with a batch data
        """
        pass

    def choose_action(self, x):
        """ Select Action according to the current state
        Args:
            x: np.array, current state
        """
        pass

    def load_model(self, model_name):
        """ Load Torch Model from files
        Args:
            model_name: str, model path
        """
        pass

    def save_model(self, model_dir, title):
        """ Save Torch Model from files
        Args:
            model_dir: str, model dir
            title: str, model name
        """
        pass


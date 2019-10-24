# -*- coding: utf-8 -*-
"""

Replay Memory Utils
"""

import os
import sys
import argparse
import numpy as np
import pickle
sys.path.append('../')


def aggravate_memories(mem_dir, save_path):
    """ aggravate the memories generated, to get only one memory file
    Args:
        mem_dir: str, memory files directory
        save_path: str, memory path of the dest memory file
    """
    files = os.listdir(mem_dir)
    data = []
    for fi in files:
        if os.path.isdir(fi):
            continue

        with open(os.path.join(mem_dir, fi), 'rb') as f:
            data += pickle.load(f)

    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print("Memory in {} have been arrgravated in {}".format(mem_dir, save_path))


def calculate_stats(mem_dir, save_path):
    """ calculate the statistics of the memory, including mean and var
    Args:
        mem_dir: str, memory file path
        save_path: str, memory path of the dest memory file
    """
    files = os.listdir(mem_dir)
    data = []
    for fi in files:
        if os.path.isdir(fi):
            continue

        with open(os.path.join(mem_dir, fi), 'rb') as f:
            data += pickle.load(f)

    states = []
    states += [x.state for x in data]
    states += [x.next_state for x in data]

    states = np.array(states)
    sample_mean = np.mean(states, axis=0)
    sample_var = np.sum(np.square((states - sample_mean)), axis=0) / states.shape[0]

    with open(save_path, 'wb') as f:
        pickle.dump((sample_mean, sample_var), f)

    print("stats (mean, var) has been calculated and saved")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='stats', type=str, help='actions: [stats, aggravate]')
    parser.add_argument('--mem_path', required=True, type=str, help='memory files dir')
    parser.add_argument('--save_path', required=True, type=str, help='save processed memory files dir')

    opt = parser.parse_args()

    if opt.phase == 'stats':
        calculate_stats(opt.mem_path, opt.save_path)
    else:
        aggravate_memories(opt.mem_path, opt.save_path)

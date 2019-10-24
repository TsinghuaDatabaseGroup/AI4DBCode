# -*- coding: utf-8 -*-
"""
description: Collect Fine State-Action pairs
"""

import os
import pickle
import argparse


def aggravate_sapairs(sa_dir, save_path):
    """ aggravate the State-Action pair files
    Args:
        sa_dir: str, files directory
        save_path: str, save path of the dest sa file
    """
    files = os.listdir(sa_dir)
    data = []
    for fi in files:
        if os.path.isdir(fi):
            continue

        with open(os.path.join(sa_dir, fi), 'rb') as f:
            data += pickle.load(f)

    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print("SA Pair in {} have been arrgravated in {}".format(sa_dir, save_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--sa_dir', required=True, type=str, help='state-action files dir')
    parser.add_argument('--save_path', required=True, type=str, help='save processed memory files dir')
    opt = parser.parse_args()
    aggravate_sapairs(opt.sa_dir, opt.save_path)


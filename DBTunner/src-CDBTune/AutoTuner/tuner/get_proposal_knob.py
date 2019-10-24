# -*- coding: utf-8 -*-
"""
description: Get the proposal knob from file
"""

import os
import sys
import utils
import pickle
import argparse
import tuner_configs
sys.path.append('../')
import environment


def get_proposal_knob(filename, gamma=0.5, idx=-1):
    assert os.path.exists(filename), "File:{} NOT EXISTS".format(filename)
    with open(filename, 'rb') as f:
        knob_data = pickle.load(f)

    max_idx = idx
    if idx == -1:
        max_score = -100
        for i in xrange(len(knob_data)):
            knob_info = knob_data[i]
            tps_inc = knob_info['tps_inc']
            lat_dec = knob_info['lat_dec']
            score = tps_inc * (1-gamma) + lat_dec * gamma

            if score > max_score:
                max_score = score
                max_idx = i

    knob_info = knob_data[max_idx]
    knob = knob_info['knob']
    metric = knob_info['metrics']
    print("[Knob] Tps: {} Latency: {}".format(metric['tps'], metric['latency']))
    return knob


def setting_knob(env, knob):
    env.setting(knob)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str, default='mysql1', help='Choose MySQL Instance')
    parser.add_argument('--knobfile', type=str, default='', help='Knob file path')
    parser.add_argument('--knobidx', type=int, default=-1, help='Proposal Knob Index in file')
    parser.add_argument('--tencent', action='store_true', help='Use Tencent Server')
    parser.add_argument('--ratio', type=float, default=0.5, help='tps versus lat ration')

    opt = parser.parse_args()
    if opt.tencent:
        env = environment.TencentServer(wk_type=opt.workload, instance_name=opt.instance,
                                        request_url=tuner_configs.TENCENT_URL)
    else:
        env = environment.Server(wk_type=opt.workload, instance_name=opt.instance)

    knob = get_proposal_knob(opt.knobfile, idx=opt.knobidx, gamma=opt.ratio)
    print("Finding Knob Finished")
    setting_knob(knob)
    print("Setting Knob Finished")


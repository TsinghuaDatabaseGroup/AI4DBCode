# -*- coding: utf-8 -*-
"""
Train the model with supervised method with saving data
"""

import os
import sys
import utils
import pickle
import random
import argparse
import tuner_configs
sys.path.append('../')
import models
import environment


parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=str, default='train', help='train or test')
parser.add_argument('--tencent', action='store_true', help='Use Tencent Server')
parser.add_argument('--params', type=str, default='', help='Load existing parameters')
parser.add_argument('--instance', type=str, default='mysql1', help='Choose MySQL Instance')
parser.add_argument('--sa_path', type=str, default='', help='state action dataset')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--epoches', type=int, default=20, help='training epoches')
parser.add_argument('--workload', type=str, default='read', help='Workload type [`read`, `write`, `readwrite`]')


opt = parser.parse_args()
print(opt)
tconfig = tuner_configs.config
ddpg_opt = dict()
ddpg_opt['tau'] = 0.01
ddpg_opt['alr'] = 0.0005
ddpg_opt['clr'] = 0.0001
ddpg_opt['model'] = opt.params
ddpg_opt['gamma'] = ""
ddpg_opt['batch_size'] = tconfig['batch_size']
ddpg_opt['memory_size'] = tconfig['memory_size']
batch_size = opt.batch_size

model = models.DDPG(n_states=tconfig['num_states'], n_actions=tconfig['num_actions'], opt=ddpg_opt, supervised=True)

if not os.path.exists('log'):
    os.mkdir('log')

if opt.phase == 'train':

    if not os.path.exists('sl_model_params'):
        os.mkdir('sl_model_params')

    expr_name = 'sl_train_ddpg_{}'.format(str(utils.get_timestamp()))

    logger = utils.Logger(
        name='ddpg',
        log_file='log/{}.log'.format(expr_name)
    )

    assert len(opt.sa_path) != 0, "SA_PATH should be specified when training DDPG Actor"

    with open(opt.sa_path, 'rb') as f:
        data = pickle.load(f)

    for epoch in xrange(opt.epoches):

        random.shuffle(data)
        num_samples = len(data)
        print(num_samples)
        n_train_samples = int(num_samples * 0.8)
        n_test_samples = num_samples - n_train_samples
        train_data = data[:n_train_samples]
        test_data = data[n_train_samples:]

        _loss = 0

        for i in xrange(n_train_samples/batch_size):

            batch_data = train_data[i*batch_size: (i+1)*batch_size]
            batch_states = [x[0].tolist() for x in batch_data]
            batch_actions = [x[1].tolist() for x in batch_data]

            _loss += model.train_actor((batch_states, batch_actions), is_train=True)

            if (i+1) % 10 == 0:
                print("[Epoch {}][Step {}] Loss: {}".format(epoch, i, _loss/(i+1)))

        test_loss = 0
        for i in xrange(n_test_samples / batch_size):
            batch_data = test_data[i * batch_size: (i + 1) * batch_size]
            batch_states = [x[0].tolist() for x in batch_data]
            batch_actions = [x[1].tolist() for x in batch_data]

            test_loss += model.train_actor((batch_states, batch_actions), is_train=False)

        print("[Epoch {}] Test Loss: {}".format(epoch, test_loss))
        model.save_actor('sl_model_params/sl_train_actor_{}.pth'.format(epoch))

else:
    # Create Environment
    if opt.tencent:
        env = environment.TencentServer(wk_type=opt.workload, instance_name=opt.instance,
                                        request_url=tuner_configs.TENCENT_URL)
    else:
        env = environment.DockerServer(wk_type=opt.workload, instance_name=opt.instance)

    current_knob = environment.get_init_knobs()

    expr_name = 'sl_test_ddpg_{}'.format(str(utils.get_timestamp()))

    logger = utils.Logger(
        name='train_supervised',
        log_file='log/{}.log'.format(expr_name)
    )

    assert len(opt.params) != 0, "Please add params' path"

    def generate_knob(action):
        return environment.gen_continuous(action)

    model.load_actor(opt.params)

    step_counter = 0

    max_step = 0
    max_value = 0.0
    generate_knobs = []
    current_state = env.initialize()
    model.reset(0.01)
    while step_counter < 20:
        state = current_state
        action = model.choose_action(state)
        current_knob = generate_knob(action)
        logger.info("[ddpg] Action: {}".format(action))

        reward, state_, done, score, metrics = env.step(current_knob)
        logger.info("[Step: {}][Metric tps:{} lat:{} qps:{}]Reward: {} Score: {} Done: {}".format(
            step_counter, metrics[0], metrics[1], metrics[2], reward, score, done
        ))
        next_state = state_

        current_state = next_state
        step_counter += 1
        generate_knobs.append((score, current_knob))
        if max_value < score:
            max_step = step_counter - 1
            max_value = score

        if done:
            break

    print("Searching Finished")
    with open(expr_name + '.pkl', 'wb') as f:
        pickle.dump(generate_knobs, f)

    print("Knobs are saved!")
    # eval

    default_konbs = environment.get_init_knobs()
    max_knobs = generate_knobs[max_step][1]

    metric1 = env.eval(default_konbs)
    print("Default TPS: {} Latency: {}".format(metric1['tps'], metric1['latency']))
    metric2 = env.eval(max_knobs)
    print("Max TPS: {} Latency: {}".format(metric2['tps'], metric2['latency']))

    delta_tps = (metric2['tps'] - metric1['tps']) / metric1['tps']
    delta_latency = (-metric2['latency'] + metric1['latency']) / metric1['latency']

    print("[Evaluation Result] Latency Decrease: {} TPS Increase: {}".format(delta_latency, delta_tps))







import filter_env
from environment import Environment
from ddpg import *
import gc
import time
gc.enable()

EPISODES = 100
TEST = 10
workloads = ['ConnectedComponent', 'PageRank', 'ShortestPaths', 'StronglyConnectedComponent',
             'LabelPropagation', 'PregelOperation', 'TriangleCount', 'SVDPlusPlus']
# workload = 'ShortestPaths'

def main():
    rewards = []
    env = Environment(workload)
    tf.reset_default_graph()
    agent = DDPG(env)

    state = env.reset()
    for episode in range(EPISODES):
        f = open('result.txt', 'a', encoding='utf-8')
        f.write("EPISODES:" + str(episode + 1) + "\n")
        f.write("cur_state:" + str(state) + "\n")
        print("EPISODES:" + str(episode + 1))
        print("cur_state:" + str(state))
        #print "episode:",episode
        # Train
        action = agent.noise_action(state)
        # print(action)
        next_state,reward,done,_ = env.step(action,f)
        rewards.append(reward)
        f.write("new_state:" + str(next_state) + "\n")
        f.write("reward:" + str(reward) + "\n")
        print("new_state:" + str(next_state))
        print("reward:" + str(reward))
        f.write("min_reward_now:" + str(max(rewards)) + "\n")
        print("min_reward_now:" + str(max(rewards)))
        agent.perceive(state,action,reward,next_state,done)
        state = next_state
        end_time = time.time()
        f.write("time_total_consume:" + str(end_time - start_time) + "\n\n")
        print("time_consume:" + str(end_time - start_time) + "\n\n")
        f.close()
        if (end_time - start_time) > 7200:
            break



if __name__ == '__main__':
    for w in range(len(workloads)):
        workload = workloads[w]
        f = open('result.txt', 'a', encoding='utf-8')
        f.write(workload + ":\n")
        print(workload + ":")
        f.close()
        start_time = time.time()
        main()

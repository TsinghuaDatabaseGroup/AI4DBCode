
from PGUtils import PGRunner
from sqlSample import sqlInfo
import numpy as np
from itertools import count
from math import log
import random
import time
from DQN import DQN,ENV
from TreeLSTM import SPINN
from JOBParser import DB
import copy
import torch
from torch.nn import init
from ImportantConfig import Config

config = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open(config.schemaFile, "r") as f:
    createSchema = "".join(f.readlines())

db_info = DB(createSchema)

featureSize = 128

policy_net = SPINN(n_classes = 1, size = featureSize, n_words = 50,mask_size= len(db_info)*len(db_info),device=device).to(device)
target_net = SPINN(n_classes = 1, size = featureSize, n_words = 50,mask_size= len(db_info)*len(db_info),device=device).to(device)
for name, param in policy_net.named_parameters():
    print(name,param.shape)
    if len(param.shape)==2:
        init.xavier_normal(param)
    else:
        init.uniform(param)

# policy_net.load_state_dict(torch.load("JOB_tc.pth"))#load cost train model
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

pgrunner = PGRunner(config.dbName,config.userName,config.password,config.ip,config.port)

DQN = DQN(policy_net,target_net,db_info,pgrunner,device)

def k_fold(input_list,k,ix = 0):
    li = len(input_list)
    kl = (li-1)//k + 1
    train = []
    validate = []
    for idx in range(li):

        if idx%k == ix:
            validate.append(input_list[idx])
        else:
            train.append(input_list[idx])
    return train,validate


def QueryLoader(QueryDir):
    def file_name(file_dir):
        import os
        L = []
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                if os.path.splitext(file)[1] == '.sql':
                    L.append(os.path.join(root, file))
        return L
    files = file_name(QueryDir)
    sql_list = []
    for filename in files:
        with open(filename, "r") as f:
            data = f.readlines()
            one_sql = "".join(data)
            sql_list.append(sqlInfo(pgrunner,one_sql,filename))
    return sql_list

def resample_sql(sql_list):
    rewards = []
    reward_sum = 0
    rewardsP = []
    mes = 0
    for sql in sql_list:
        #         sql = val_list[i_episode%len(train_list)]
        pg_cost = sql.getDPlantecy()
        #         continue
        env = ENV(sql,db_info,pgrunner,device)

        for t in count():
            action_list, chosen_action,all_action = DQN.select_action(env,need_random=False)

            left = chosen_action[0]
            right = chosen_action[1]
            env.takeAction(left,right)

            reward, done = env.reward()
            if done:
                mrc = max(np.exp(reward*log(1.5))/pg_cost-1,0)
                rewardsP.append(np.exp(reward*log(1.5)-log(pg_cost)))
                mes += reward*log(1.5)-log(pg_cost)
                rewards.append((mrc,sql))
                reward_sum += mrc
                break
    import random
    print(rewardsP)
    res_sql = []
    print(mes/len(sql_list))
    for idx in range(len(sql_list)):
        rd = random.random()*reward_sum
        for ts in range(len(sql_list)):
            rd -= rewards[ts][0]
            if rd<0:
                res_sql.append(rewards[ts][1])
                break
    return res_sql+sql_list
def train(trainSet,validateSet):

    trainSet_temp = trainSet
    losses = []
    startTime = time.time()
    print_every = 20
    TARGET_UPDATE = 3
    for i_episode in range(0,10000):
        if i_episode % 200 == 100:
            trainSet = resample_sql(trainSet_temp)
        #     sql = random.sample(train_list_back,1)[0][0]
        sqlt = random.sample(trainSet[0:],1)[0]
        pg_cost = sqlt.getDPlantecy()
        env = ENV(sqlt,db_info,pgrunner,device)

        previous_state_list = []
        action_this_epi = []
        nr = True
        nr = random.random()>0.3 or sqlt.getBestOrder()==None
        acBest = (not nr) and random.random()>0.7
        for t in count():
            # beginTime = time.time();
            action_list, chosen_action,all_action = DQN.select_action(env,need_random=nr)
            value_now = env.selectValue(policy_net)
            next_value = torch.min(action_list).detach()
            # e1Time = time.time()
            env_now = copy.deepcopy(env)
            # endTime = time.time()
            # print("make",endTime-startTime,endTime-e1Time)
            if acBest:
                chosen_action = sqlt.getBestOrder()[t]
            left = chosen_action[0]
            right = chosen_action[1]
            env.takeAction(left,right)
            action_this_epi.append((left,right))

            reward, done = env.reward()
            reward = torch.tensor([reward], device=device, dtype = torch.float32).view(-1,1)

            previous_state_list.append((value_now,next_value.view(-1,1),env_now))
            if done:

                #             print("done")
                next_value = 0
                sqlt.updateBestOrder(reward.item(),action_this_epi)

            expected_state_action_values = (next_value ) + reward.detach()
            final_state_value = (next_value ) + reward.detach()

            if done:
                cnt = 0
                #             for idx in range(t-cnt+1):
                global tree_lstm_memory
                tree_lstm_memory = {}
                DQN.Memory.push(env,expected_state_action_values,final_state_value)
                for pair_s_v in previous_state_list[:0:-1]:
                    cnt += 1
                    if expected_state_action_values > pair_s_v[1]:
                        expected_state_action_values = pair_s_v[1]
                    #                 for idx in range(cnt):
                    expected_state_action_values = expected_state_action_values
                    DQN.Memory.push(pair_s_v[2],expected_state_action_values,final_state_value)
                #                 break
                loss = 0

            if done:
                # break
                loss = DQN.optimize_model()
                loss = DQN.optimize_model()
                loss = DQN.optimize_model()
                loss = DQN.optimize_model()
                losses.append(loss)
                if ((i_episode + 1)%print_every==0):
                    print(np.mean(losses))
                    print("######################Epoch",i_episode//print_every,pg_cost)
                    val_value = DQN.validate(validateSet)
                    print("time",time.time()-startTime)
                    print("~~~~~~~~~~~~~~")
                break
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

if __name__=='__main__':
    sytheticQueries = QueryLoader(QueryDir=config.sytheticDir)
    # print(sytheticQueries)
    JOBQueries = QueryLoader(QueryDir=config.JOBDir)
    Q4,Q1 = k_fold(JOBQueries,10,1)
    # print(Q4,Q1)
    train(Q4+sytheticQueries,Q1)

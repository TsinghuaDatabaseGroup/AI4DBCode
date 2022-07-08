# Copyright 2018-2021 Tsinghua DBGroup
#
# Licensed under the Apache License, Version 2.0 (the "License"): you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
import math
import random
import torchfold
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as T
import torch
from collections import namedtuple
from sqlSample import JoinTree
import torch.optim as optim
import numpy as np
from math import log
from itertools import count
from PGUtils import pgrunner

from ImportantConfig import Config
# def Floss(value,targetvalue):
#     with torch.no_grad():
#         disl1 = (torch.abs(value-targetvalue)<1).float()
#     return torch.mean((1-disl1)*torch.abs(value-targetvalue)*(targetvalue+1)+disl1*(value-targetvalue)*(value-targetvalue))
def Floss(value,targetvalue):
    with torch.no_grad():
        disl1 = (torch.abs(value-targetvalue)<0.15).float()
    with torch.no_grad():
        disl2 = 1-((value>targetvalue).float()*(targetvalue>config.maxR-0.1).float())
    
    return torch.mean(disl2*((1-disl1)*torch.abs(value-targetvalue)*(targetvalue+1)+disl1*(value-targetvalue)*(value-targetvalue)))
config = Config()
class ENV(object):
    def __init__(self,sql,db_info,pgrunner,device,run_mode = False):
        self.sel = JoinTree(sql,db_info )
        self.sql = sql
        self.hashs = ""
        self.table_set = set([])
        self.res_table = []
        self.init_table = None
        self.planSpace = 0#0:leftDeep,1:bushy
        self.run_mode = run_mode

    def actionValue(self,left,right,model):
        self.sel.joinTables(left,right,fake = True)
        res_Value = self.selectValue(model)
        self.sel.total -= 1
        self.sel.aliasnames_root_set.remove(self.sel.total)
        self.sel.aliasnames_fa.pop(self.sel.left_son[self.sel.total])
        self.sel.aliasnames_fa.pop(self.sel.right_son[self.sel.total])
        return res_Value

    def selectValue(self,model):
        tree_state = []
        # print("--------")
        for idx in self.sel.aliasnames_root_set:
            # print(idx,idx in self.sel.aliasnames_fa,isinstance(idx,int))
            # if (not idx in self.sel.aliasnames_fa )and isinstance(idx,int):
            if (not idx in self.sel.aliasnames_fa ):
                tree_state.append(self.sel.encode_tree_regular(model,idx))
        # if len(tree_state)!=1:
        #     print(-1)
        res = torch.cat(tree_state,dim = 0)
        # if len(self.sel.aliasnames_root_set)==len(self.join_list)+1:
        #     print(res)
        return model.logits(res,self.sel.join_matrix,len(self.sel.aliasnames_root_set)==len(self.sel.join_list)+1)

    def selectValueFold(self,fold):
        tree_state = []
        for idx in self.sel.aliasnames_root_set:
            # if (not idx in self.sel.aliasnames_fa )and isinstance(idx,int):
            if (not idx in self.sel.aliasnames_fa):
                tree_state.append(self.sel.encode_tree_fold(fold,idx))
            #         res = torch.cat(tree_state,dim = 0)
        return tree_state
        return fold.add('logits',tree_state,self.sel.join_matrix)



    def takeAction(self,left,right):
        self.sel.joinTables(left,right)
        self.hashs += left
        self.hashs += right
        self.hashs += " "

    def hashcode(self):
        return self.sql.sql+self.hashs
    def allAction(self,model):
        action_value_list = []
        for one_join in self.sel.join_candidate:
            l_fa = self.sel.findFather(one_join[0])
            r_fa  =self.sel.findFather(one_join[1])
            if self.planSpace ==0:
                flag1 = one_join[1] ==r_fa and l_fa !=one_join[0]
                if l_fa!=r_fa and (self.sel.total == 0 or flag1):
                    action_value_list.append((self.actionValue(one_join[0],one_join[1],model),one_join))
            elif self.planSpace==1:
                if l_fa!=r_fa:
                    action_value_list.append((self.actionValue(one_join[0],one_join[1],model),one_join))
        return action_value_list
    def reward_new(self,):
        from math import e
        if self.sel.total+1 == len(self.sel.from_table_list):
            if self.run_mode == False:
                return self.sel.plan2Cost()/self.sel.sqlt.getDPlantecy(), True
            else:
                return self.sel.getResult(),True
        else:
            return 0,False



Transition = namedtuple('Transition',
                        ('env', 'next_value', 'this_value'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.bestJoinTreeValue = {}
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        data =  Transition(*args)
        hashv = data.env.hashcode()
        next_value = data.next_value
        this_value = data.this_value
        if hashv in self.bestJoinTreeValue and self.bestJoinTreeValue[hashv]<data.this_value:
            if self.bestJoinTreeValue[hashv]<next_value:
                next_value = self.bestJoinTreeValue[hashv]
            this_value = self.bestJoinTreeValue[hashv]
            # import random
            # if (data.this_value < self.bestJoinTreeValue[hashv]+0.3 and random.random()>0.3):
            #     return
        else:
            self.bestJoinTreeValue[hashv]  = data.this_value
        # from math import e
        # if next_value<22220:
        # print(self.bestJoinTreeValue[hashv])
        data = Transition(data.env,self.bestJoinTreeValue[hashv],hashv)
        # else:
        #     data = Transition(data.env,next_value,this_value)
        position = self.position
        self.memory[position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory)>batch_size:
            return random.sample(self.memory, batch_size)
        else:
            return self.memory

    def __len__(self):
        return len(self.memory)
    def resetMemory(self,):
        self.memory =[]
        self.position = 0
    def resetbest(self):
        self.bestJoinTreeValue = {}

class DQN:
    def __init__(self,policy_net,target_net,db_info,pgrunner,device):
        self.Memory = ReplayMemory(config.memory_size)
        self.BATCH_SIZE = 1

        self.optimizer = optim.Adam(policy_net.parameters(),lr = 3e-4   ,betas=(0.9,0.999))
        # self.optimizer = optim.SGD(policy_net.parameters(),lr = config.learning_rate)

        self.steps_done = 0
        self.max_action = 25
        # self.EPS_START = 0.8
        # self.EPS_END = 0.1
        # self.EPS_DECAY = 50
        self.EPS_START = config.EPS_START
        self.EPS_END =  config.EPS_END
        self.EPS_DECAY = config.EPS_DECAY
        self.policy_net = policy_net
        self.target_net = target_net
        self.db_info = db_info
        self.device = device
        self.steps_done = 0
    def select_action(self, env, need_random = True):

        sample = random.random()
        if need_random:
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                                      math.exp(-1. * self.steps_done / self.EPS_DECAY)
            self.steps_done += 1
        else:
            eps_threshold = -1
        action_list = env.allAction(self.policy_net)
        action_batch = torch.cat([x[0] for x in action_list],dim = 1)
        # print('select_action',sample > eps_threshold)
        # action_list
        if sample > eps_threshold:
            return action_batch,action_list[torch.argmin(action_batch,dim = 1)[0]][1],[x[1] for x in action_list]
        else:
            return action_batch,action_list[random.randint(0,len(action_list)-1)][1],[x[1] for x in action_list]


    def validate(self,val_list, tryTimes = 1):
        rewards = []
        prt = []
        mes = 0
        val_this_time = {}
        DP_cost = 0.0
        my_cost = 0.0
        allRes = {}
        import time
        startTime = time.time()
        valInfo  = {}
        for sql in val_list:
            # print('DQN')
            pg_cost = sql.getDPlantecy()
            # print('outPG_cost')
            DP_cost += pg_cost
            env = ENV(sql,self.db_info,pgrunner,self.device)

            for t in count():

                action_list, chosen_action,all_action = self.select_action(env,need_random=False)

                left = chosen_action[0]
                right = chosen_action[1]
                env.takeAction(left,right)

                startTime = time.time()
                reward, done = env.reward_new()
                thisTime = time.time() -startTime
                lalias = len(env.sel.from_table_list)
                if done:
                    val_this_time[sql.filename] = reward
                    allRes[sql.filename] = (pg_cost,reward*pg_cost)
                    rewards.append(reward)
                    mes = mes + log(reward)
                    my_cost += reward*pg_cost
                    valInfo[sql.filename] = (lalias,thisTime,reward*pg_cost,pg_cost)
                    break
        import json
        lr = len(rewards)
        from math import e
        print("MRC",sum(rewards)/lr,"GMRL",e**(mes/lr),"SMRC",my_cost/DP_cost)
        return sum(rewards)/lr,e**(mes/lr),my_cost/DP_cost
    def validate_ind(self,val_list, tryTimes = 1):
        rewards = []
        prt = []
        mes = 0
        for sql in val_list:
            pg_cost = sql.getDPlantecy()
            env = ENV(sql,self.db_info,pgrunner,self.device)
            if (len(env.sel.from_table_list)<3):# or not env.sel.baseline.left_deep or (sql.bestLatency is not None and sql.bestLatency>config.baselineValue)) and not config.testGen:
                    rewards.append(-1.0)
                    continue
            if (len(env.sel.from_table_list)<3):
                rewards.append(-1.0)
                continue
            if (sql.useCost):
                rewards.append(-1.0)
                print('useCost')
                continue
            for t in count():
                action_list, chosen_action,all_action = self.select_action(env,need_random=False)
                left = chosen_action[0]
                right = chosen_action[1]
                env.takeAction(left,right)

                reward, done = env.reward_new()
                if done:
                    rewards.append(reward)
                    break
        return rewards
    def optimize_model(self,):
        import time
        startTime = time.time()
        samples = self.Memory.sample(128)
        value_now_list = []
        next_value_list = []
        # if (len(samples)<64):
        #     return 0
        usecuda = True if config.usegpu == 1 else False
        # print(torchfold.path)
        fold = torchfold.Fold(cuda=usecuda)
        nowL = []
        for one_sample in samples:
            nowList = one_sample.env.selectValueFold(fold)
            nowL.append(len(nowList))
            value_now_list+=nowList
        res = fold.apply(self.policy_net, [value_now_list])[0]
        total = 0
        value_now_list = []
        next_value_list = []
        for idx,one_sample in enumerate(samples):
            value_now_list.append(self.policy_net.logits(res[total:total+nowL[idx]] , one_sample.env.sel.join_matrix ))
            next_value_list.append(min(one_sample.next_value,self.Memory.bestJoinTreeValue[one_sample.this_value]))
            total += nowL[idx]
        value_now = torch.cat(value_now_list,dim = 0)
        next_value = torch.cat(next_value_list,dim = 0)
        endTime = time.time()
        if True:
            loss = Floss(value_now,next_value)
            # loss = F.smooth_l1_loss(value_now,next_value,size_average=True)
            self.optimizer.zero_grad()
            loss.backward()
            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            return loss.item()
        return None

        
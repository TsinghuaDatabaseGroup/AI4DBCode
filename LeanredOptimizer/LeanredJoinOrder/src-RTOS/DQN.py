
import math
import random
import torchfold
import torch.nn.functional as F
import torchvision.transforms as T
import torch
from collections import namedtuple
from sqlSample import JoinTree
import torch.optim as optim
import numpy as np
from math import log
from itertools import count

class ENV(object):
    def __init__(self,sql,db_info,pgrunner,device):
        self.sel = JoinTree(sql,db_info,pgrunner,device )
        self.sql = sql
        self.hashs = ""
        self.table_set = set([])
        self.res_table = []
        self.init_table = None
        self.planSpace = 0#0:leftDeep,1:bushy

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
        for idx in self.sel.aliasnames_root_set:
            if not idx in self.sel.aliasnames_fa:
                tree_state.append(self.sel.encode_tree_regular(model,idx))
        res = torch.cat(tree_state,dim = 0)
        return model.logits(res,self.sel.join_matrix)

    def selectValueFold(self,fold):
        tree_state = []
        for idx in self.sel.aliasnames_root_set:
            if not idx in self.sel.aliasnames_fa:
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
    def reward(self,):
        if self.sel.total+1 == len(self.sel.from_table_list):
            return log( self.sel.plan2Cost())/log(1.5), True
        else:
            return 0,False



Transition = namedtuple('Transition',
                        ('env', 'next_value', 'this_value'))
# bestJoinTreeValue = {}
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
        if hashv in self.bestJoinTreeValue and self.bestJoinTreeValue[hashv]<data.this_value:
            if self.bestJoinTreeValue[hashv]<next_value:
                next_value = self.bestJoinTreeValue[hashv]
        else:
            self.bestJoinTreeValue[hashv]  = data.this_value
        data = Transition(data.env,self.bestJoinTreeValue[hashv],data.this_value)
        position = self.position
        self.memory[position] = data
        #         self.position
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
    def resetbest(self):
        self.bestJoinTreeValue = {}

class DQN:
    def __init__(self,policy_net,target_net,db_info,pgrunner,device):
        self.Memory = ReplayMemory(1000)
        self.BATCH_SIZE = 1

        self.optimizer = optim.Adam(policy_net.parameters(),lr = 3e-4   ,betas=(0.9,0.999))

        self.steps_done = 0
        self.max_action = 25
        self.EPS_START = 0.4
        self.EPS_END = 0.2
        self.EPS_DECAY = 400
        self.policy_net = policy_net
        self.target_net = target_net
        self.db_info = db_info
        self.pgrunner = pgrunner
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

        if sample > eps_threshold:
            return action_batch,action_list[torch.argmin(action_batch,dim = 1)[0]][1],[x[1] for x in action_list]
        else:
            return action_batch,action_list[random.randint(0,len(action_list)-1)][1],[x[1] for x in action_list]


    def validate(self,val_list, tryTimes = 1):
        rewards = []
        prt = []
        mes = 0
        for sql in val_list:
            pg_cost = sql.getDPlantecy()
            env = ENV(sql,self.db_info,self.pgrunner,self.device)

            for t in count():
                action_list, chosen_action,all_action = self.select_action(env,need_random=False)

                left = chosen_action[0]
                right = chosen_action[1]
                env.takeAction(left,right)

                reward, done = env.reward()
                if done:
                    rewards.append(np.exp(reward*log(1.5)-log(pg_cost)))
                    mes = mes + reward*log(1.5)-log(pg_cost)
                    break
        lr = len(rewards)
        from math import e
        print("MRC",sum(rewards)/lr,"GMRL",e**(mes/lr))
        return sum(rewards)/lr

    def optimize_model(self,):
        import time
        startTime = time.time()
        samples = self.Memory.sample(64)
        value_now_list = []
        next_value_list = []
        if (len(samples)==0):
            return
        fold = torchfold.Fold(cuda=True)
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
            next_value_list.append(one_sample.next_value)
            total += nowL[idx]
        value_now = torch.cat(value_now_list,dim = 0)
        next_value = torch.cat(next_value_list,dim = 0)
        endTime = time.time()
        if True:
            loss = F.smooth_l1_loss(value_now,next_value,size_average=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item()
        return None
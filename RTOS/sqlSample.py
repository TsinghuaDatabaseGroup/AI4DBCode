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

import sys
sys.path.append(".")

from JOBParser import TargetTable,FromTable,Comparison
max_column_in_table = 15
import torch
import torch
import torch.nn as nn
from itertools import count
import numpy as np
from PGUtils import pgrunner

from ImportantConfig import Config

config = Config()

device = torch.device("cuda" if torch.cuda.is_available() and config.usegpu==1 else "cpu")

class BaselineAlias:
    def __init__(self,rows00,alias):
        self.rows00 = rows00
        self.alias = alias
        self.result_order = []
        self.joinedset = set()
        self.left_deep = True
        if rows00[0][0][0]['Plan']['Total Cost']>1:
            self.getBaseline(rows00[0][0][0]['Plan'],alias)
        else:
            alist = list(self.alias)
            for x in range(1,len(self.alias)):
                self.result_order.append((alist[0],alist[x]))
        # print(rows00[0][0][0])
        if self.left_deep:
            if len(self.result_order)!=len(self.alias)-1:
                print("wrong number of order",self.result_order)
                while (1):
                    pass
    def hashAdd(self,x):
        if x.find(' AND ')>-1:
            y = x.split(' AND ')
            self.hashAdd(y[0][1:])
            self.hashAdd(y[1][0:])
        else:
            thisTime = []
            for y in self.alias:
                if x.find('('+y+'.')>-1 and x.find('('+y+'.')<2:
                    thisTime.append(y)
            
            for y in self.alias:
                if x.find(' '+y+'.')>-1 and x.find(' '+y+'.')>2:
                    thisTime.append(y)
            
            if len(thisTime)!=2:
                print(thisTime)
                print('wroing Hash')
                while (1):
                    pass
            if len(self.joinedset)!=0 and not (thisTime[0] in self.joinedset or thisTime[1] in self.joinedset):
                self.left_deep = False
            if not (thisTime[0] in self.joinedset and thisTime[1] in self.joinedset):
                if thisTime[0] in self.joinedset:
                    self.result_order.append(thisTime)
                else:
                    self.result_order.append([thisTime[1],thisTime[0]])
                self.joinedset.add(thisTime[0])
                self.joinedset.add(thisTime[1])
        
    def getBaseline(self,rows00,alias):
        import json
        # print(json.dumps(rows00))
        # print("------")
        if 'Plans' in rows00:
            for x in rows00['Plans']:
                self.getBaseline(x,alias)
        if 'Recheck Cond' in rows00 and 'Alias' in rows00 and rows00['Recheck Cond'].find(" = ")!=-1:
            thisTime = [rows00['Alias']]
            if rows00['Recheck Cond']:
                for y in self.alias:
                    if rows00['Recheck Cond'].find('(' +y+'.')!=-1 or rows00['Recheck Cond'].find(' ' +y+'.')!=-1:
                        thisTime.append(y)
            if len(self.joinedset)!=0 and not (thisTime[0] in self.joinedset or thisTime[1] in self.joinedset):
                self.left_deep = False
            if len(thisTime)!=2:
                return 
                print('wrong recheck')
                while (1):
                    pass
            if not (thisTime[0] in self.joinedset and thisTime[1] in self.joinedset):
                if len(self.joinedset)!=0 and not (thisTime[0] in self.joinedset or thisTime[1] in self.joinedset):
                    self.left_deep = False
                if thisTime[0] in self.joinedset:
                    self.result_order.append(thisTime)
                else:
                    self.result_order.append([thisTime[1],thisTime[0]])
                self.joinedset.add(thisTime[0])
                self.joinedset.add(thisTime[1])
        if 'Index Cond' in rows00 and 'Alias' in rows00 and rows00['Index Cond'].find(" = ")!=-1:
            # print(rows00['Index Cond'].find(" = ")!='-1')
            thisTime = [rows00['Alias']]
            if rows00['Index Cond']:
                for y in self.alias:
                    if rows00['Index Cond'].find('(' +y+'.')!=-1 or rows00['Index Cond'].find(' ' +y+'.')!=-1:
                        thisTime.append(y)

            if len(thisTime)!=2:
                return
                print(rows00['Index Cond'])
                print('wroing index',)
                while (1):
                    pass
            if len(self.joinedset)!=0 and not (thisTime[0] in self.joinedset or thisTime[1] in self.joinedset):
                self.left_deep = False
            if not (thisTime[0] in self.joinedset and thisTime[1] in self.joinedset):

                if len(self.joinedset)!=0 and not (thisTime[0] in self.joinedset or thisTime[1] in self.joinedset):
                    self.left_deep = False
                if thisTime[0] in self.joinedset:
                    self.result_order.append(thisTime)
                else:
                    self.result_order.append([thisTime[1],thisTime[0]])
                self.joinedset.add(thisTime[0])
                self.joinedset.add(thisTime[1])
        if 'Hash Cond' in rows00:
            self.hashAdd(rows00['Hash Cond'])
        if 'Merge Cond' in rows00:
            self.hashAdd(rows00['Merge Cond'])
        if 'Join Filter' in rows00:
            self.hashAdd(rows00['Join Filter'])
tlm = {}

class JoinTree:
    def __init__(self,sqlt,db_info,max_column_in_table = 15,max_alias = 40,extent_sql = True):
        from psqlparse import parse_dict
        global tlm
        tlm = {}
        # print(sqlt.sql)
        self.sqlt = sqlt
        self.sql = self.sqlt.sql
        # print([self.sql])
        parse_result = parse_dict(self.sql)[0]["SelectStmt"]
        self.target_table_list = [TargetTable(x["ResTarget"]) for x in parse_result["targetList"]]
        self.from_table_list = [FromTable(x["RangeVar"]) for x in parse_result["fromClause"]]
        if len(self.from_table_list)<2:
            return
        self.aliasname2fullname = {}
        # self.pgrunner = pgRunner
        self.id2aliasname = {0: 'start', 1: 'chn', 2: 'ci', 3: 'cn', 4: 'ct', 5: 'mc', 6: 'rt', 7: 't', 8: 'k', 9: 'lt', 10: 'mk', 11: 'ml', 12: 'it1', 13: 'it2', 14: 'mi', 15: 'mi_idx', 16: 'it', 17: 'kt', 18: 'miidx', 19: 'at', 20: 'an', 21: 'n', 22: 'cc', 23: 'cct1', 24: 'cct2', 25: 'it3', 26: 'pi', 27: 't1', 28: 't2', 29: 'cn1', 30: 'cn2', 31: 'kt1', 32: 'kt2', 33: 'mc1', 34: 'mc2', 35: 'mi_idx1', 36: 'mi_idx2', 37: 'an1', 38: 'n1', 39: 'a1'}
        self.aliasname2id = {'kt1': 31, 'chn': 1, 'cn1': 29, 'mi_idx2': 36, 'cct1': 23, 'n': 21, 'a1': 39, 'kt2': 32, 'miidx': 18, 'it': 16, 'mi_idx1': 35, 'kt': 17, 'lt': 9, 'ci': 2, 't': 7, 'k': 8, 'start': 0, 'ml': 11, 'ct': 4, 't2': 28, 'rt': 6, 'it2': 13, 'an1': 37, 'at': 19, 'mc2': 34, 'pi': 26, 'mc': 5, 'mi_idx': 15, 'n1': 38, 'cn2': 30, 'mi': 14, 'it1': 12, 'cc': 22, 'cct2': 24, 'an': 20, 'mk': 10, 'cn': 3, 'it3': 25, 't1': 27, 'mc1': 33}
        self.alias_selectivity = [0]*len(self.id2aliasname)
        # self.device = device
        self.aliasname2fromtable = {}
        for table in self.from_table_list:
            self.aliasname2fromtable[table.getAliasName()] = table
            self.aliasname2fullname[table.getAliasName()] = table.getFullName()
        self.aliasnames = set(self.aliasname2fromtable.keys())
        self.comparison_list =[Comparison(x) for x in parse_result["whereClause"]["BoolExpr"]["args"]]
        # print('lc',len(self.comparison_list))
        self.db_info = db_info
        self.join_list = {}
        self.filter_list = {}

        self.aliasnames_root_set = set([x.getAliasName() for x in self.from_table_list])
        self.aliasnames_fa = {}
        self.aliasnames_set = {}
        self.aliasnames_join_set = {}
        self.left_son = {}
        self.right_son = {}
        self.total = 0
        self.left_aliasname = {}
        self.right_aliasname = {}

        self.table_fea_set = {}
        for aliasname in self.aliasnames_root_set:
            self.table_fea_set[aliasname] = [0,0.0]*max_column_in_table+[1.0]

        ##提取所有的Join和filter
        self.join_candidate = set()
        self.join_matrix=[]
        for aliasname in self.aliasnames_root_set:
            self.join_list[aliasname] = []
        for idx in range(max_alias):
            self.join_matrix.append([0]*max_alias)
        if not extent_sql:
            return 
        for comparison in self.comparison_list:
            # print(str(comparison),len(comparison.aliasname_list))
            if len(comparison.aliasname_list) == 2:
                if not comparison.aliasname_list[0] in self.join_list:
                    self.join_list[comparison.aliasname_list[0]] = []
                if not comparison.aliasname_list[1] in self.join_list:
                    self.join_list[comparison.aliasname_list[1]] = []
                self.join_list[comparison.aliasname_list[0]].append((comparison.aliasname_list[1],comparison))
                left_aliasname = comparison.aliasname_list[0]
                left_fullname = self.aliasname2fullname[left_aliasname]
                left_table_class = db_info.name2table[left_fullname]
                table_idx = left_table_class.column2idx[comparison.column_list[0]]
                self.table_fea_set[left_aliasname][table_idx * 2] = 1
                self.join_list[comparison.aliasname_list[1]].append((comparison.aliasname_list[0],comparison))
                right_aliasname = comparison.aliasname_list[1]
                right_fullname = self.aliasname2fullname[right_aliasname]
                right_table_class = db_info.name2table[right_fullname]
                table_idx = right_table_class.column2idx[comparison.column_list[1]]
                self.table_fea_set[right_aliasname][table_idx * 2] = 1
                self.join_candidate.add((comparison.aliasname_list[0],comparison.aliasname_list[1]))
                self.join_candidate.add((comparison.aliasname_list[1],comparison.aliasname_list[0]))
                # print('111',comparison,comparison.aliasname_list[0],comparison.aliasname_list[1])
                # idx0 = self.db_info.name2idx[left_fullname]
                # idx1 = self.db_info.name2idx[right_fullname]
                idx0 = self.aliasname2id[left_aliasname]
                idx1 = self.aliasname2id[right_aliasname]
                self.join_matrix[idx0][idx1] = 1
                self.join_matrix[idx1][idx0] = 1
            else:
                if not comparison.aliasname_list[0] in self.filter_list:
                    self.filter_list[comparison.aliasname_list[0]] = []
                self.filter_list[comparison.aliasname_list[0]].append(comparison)
                left_aliasname = comparison.aliasname_list[0]
                left_fullname = self.aliasname2fullname[left_aliasname]
                left_table_class = db_info.name2table[left_fullname]
                table_idx = left_table_class.column2idx[comparison.column_list[0]]
                self.table_fea_set[left_aliasname][table_idx * 2 + 1] = max(self.table_fea_set[left_aliasname][table_idx * 2 + 1],pgrunner.getSelectivity(str(self.aliasname2fromtable[comparison.aliasname_list[0]]),str(comparison)))
                self.alias_selectivity[self.aliasname2id[left_aliasname]] = max(self.alias_selectivity[self.aliasname2id[left_aliasname]],pgrunner.getSelectivity(str(self.aliasname2fromtable[comparison.aliasname_list[0]]),str(comparison)))

        for aliasname in self.aliasnames_root_set:
            self.table_fea_set[aliasname] = torch.tensor(self.table_fea_set[aliasname],device = device).reshape(1,-1).detach()
            self.aliasnames_set[aliasname] = set([aliasname])
            for y in self.join_list[aliasname]:
                if aliasname not in self.aliasnames_join_set:
                    self.aliasnames_join_set[aliasname] = set()
                self.aliasnames_join_set[aliasname].add(y[0])


        predice_list_dict={}
        # for table in self.db_info.tables:
        #     predice_list_dict[table.name] = [0] * len(table.column2idx)
        table_dict = {}
        for table in self.db_info.tables:
            table_dict[table.name] = table 
        for alias in self.aliasname2id:
            predice_list_dict[alias] = [0] * len(table.column2idx)
        for filter_table in self.filter_list:
            for comparison in self.filter_list[filter_table]:
                aliasname = comparison.aliasname_list[0]
                fullname = self.aliasname2fullname[aliasname]
                table = self.db_info.name2table[fullname]
                for column in comparison.column_list:
                    columnidx = table.column2idx[column]
                    predice_list_dict[aliasname][columnidx] = 1
        # self.predice_feature = []
        # for fullname in predice_list_dict:
        #     self.predice_feature+= predice_list_dict[fullname]
        # # print('all_join',self.join_candidate)
        # self.predice_feature = np.asarray(self.predice_feature).reshape(1,-1)
        self.join_matrix = torch.tensor(np.concatenate((np.asarray(self.join_matrix).reshape(1,-1),np.asarray(self.alias_selectivity).reshape(1,-1)),axis=1),device = device,dtype = torch.float32)
        # print("----begin-----")
        # print(self.sql)
        import json
        # print(json.dumps(pgRunner.getPlan(self.sql)))
        # print("fck baseline")
        self.baseline = BaselineAlias(pgrunner.getPlan(self.sql),self.aliasnames)
        # print("-----end-----")


    def comparisonExpand(self,comparison_list):
        join_matrix = {}
        # for idx in range(len(self.db_info)):
        #     join_matrix.append([0]*len(self.db_info))
        for comparison in comparison_list:
            if len(comparison.aliasname_list) == 2:
                left_aliasname = comparison.aliasname_list[0]
                left_fullname = self.aliasname2fullname[left_aliasname]
                left_table_class = self.db_info.name2table[left_fullname]
                table_idx = left_table_class.column2idx[comparison.column_list[0]]
                right_aliasname = comparison.aliasname_list[1]
                right_fullname = self.aliasname2fullname[right_aliasname]
                right_table_class = self.db_info.name2table[right_fullname]
                table_idx = right_table_class.column2idx[comparison.column_list[1]]
                idx0 = self.db_info.name2idx[left_fullname]
                idx1 = self.db_info.name2idx[right_fullname]
                join_matrix[(left_aliasname,right_aliasname)] = 1
                join_matrix[(right_aliasname,left_aliasname)] = 1
        Flag = True
        while Flag:
            newList = []
            Flag = False
            for comparison1 in comparison_list:
                if len(comparison1.aliasname_list) == 2:
                    left_aliasname1 = comparison1.aliasname_list[0]
                    left_fullname1 = self.aliasname2fullname[left_aliasname1]
                    left_columnname1 = comparison1.column_list[0]
                    right_aliasname1 = comparison1.aliasname_list[1]
                    right_fullname1 = self.aliasname2fullname[right_aliasname1]
                    right_columnname1 = comparison1.column_list[1]
                    idx1l = self.db_info.name2idx[left_fullname1]
                    idx1r = self.db_info.name2idx[right_fullname1]
                    for comparison2 in comparison_list:
                        if len(comparison2.aliasname_list) == 2 and str(comparison1)!=str(comparison2):
                            left_aliasname2 = comparison2.aliasname_list[0]
                            left_fullname2 = self.aliasname2fullname[left_aliasname2]
                            left_columnname2 = comparison2.column_list[0]
                            right_aliasname2 = comparison2.aliasname_list[1]
                            right_fullname2 = self.aliasname2fullname[right_aliasname2]
                            right_columnname2 = comparison2.column_list[1]
                            idx2l = self.db_info.name2idx[left_fullname2]
                            idx2r = self.db_info.name2idx[right_fullname2]
                            import copy
                            if left_aliasname1 == left_aliasname2 and left_columnname1 == left_columnname2:
                                if not (right_aliasname1,right_aliasname2) in join_matrix:
                                    Flag = True
                                    # join_matrix[idx1r][idx2r]=1
                                    ncp = copy.deepcopy(comparison1)
                                    ncp.lexpr = comparison2.rexpr
                                    ncp.column_list[0] = comparison2.column_list[1]
                                    ncp.aliasname_list[0] = comparison2.aliasname_list[1]
                                    join_matrix[(right_aliasname1,right_aliasname2)] = 1
                                    join_matrix[(right_aliasname2,right_aliasname1)] = 1
                                    newList.append(ncp)
                            if left_aliasname1 == right_aliasname2 and left_columnname1 == right_columnname2:
                                if not (right_aliasname1,left_aliasname2) in join_matrix:
                                    Flag = True
                                    # join_matrix[idx1r][idx2l]=1
                                    ncp = copy.deepcopy(comparison1)
                                    ncp.lexpr = comparison2.lexpr
                                    ncp.column_list[0] = comparison2.column_list[0]
                                    ncp.aliasname_list[0] = comparison2.aliasname_list[0]
                                    join_matrix[(right_aliasname1,left_aliasname2)] = 1
                                    join_matrix[(left_aliasname2,right_aliasname1)] = 1
                                    newList.append(ncp)
                            if right_aliasname1 == right_aliasname2 and right_columnname1 == right_columnname2:
                                if not (left_aliasname1,left_aliasname2) in join_matrix:
                                    Flag = True
                                    # join_matrix[idx1l][idx2l]=1
                                    ncp = copy.deepcopy(comparison1)
                                    ncp.rexpr = comparison2.lexpr
                                    ncp.column_list[1] = comparison2.column_list[0]
                                    ncp.aliasname_list[1] = comparison2.aliasname_list[0]
                                    join_matrix[(left_aliasname1,left_aliasname2)] = 1
                                    join_matrix[(left_aliasname2,left_aliasname1)] = 1
                                    newList.append(ncp)
                            # print(len(newList),idx1r,idx2l)
            # print('sqlSample.py newList :',[newList])
            comparison_list =  comparison_list+newList
        # print("add----")
        # for cp in newList:
        #     print(str(cp))
        # print("add----")
        return comparison_list
    def resetJoin(self):
        self.aliasnames_fa = {}
        self.left_son = {}
        self.right_son = {}
        self.aliasnames_root_set = set([x.getAliasName() for x in self.from_table_list])

        self.left_aliasname  = {}
        self.right_aliasname =  {}
        self.aliasnames_join_set = {}
        for aliasname in self.aliasnames_root_set:
            self.aliasnames_set[aliasname] = set([aliasname])
            for y in self.join_list[aliasname]:
                if aliasname not in self.aliasnames_join_set:
                    self.aliasnames_join_set[aliasname] = set()
                self.aliasnames_join_set[aliasname].add(y[0])

        self.total = 0
    def findFather(self,node_name):
        fa_name = node_name
        while  fa_name in self.aliasnames_fa:
            fa_name = self.aliasnames_fa[fa_name]
        while  node_name in self.aliasnames_fa:
            temp_name = self.aliasnames_fa[node_name]
            self.aliasnames_fa[node_name] = fa_name
            node_name = temp_name
        return fa_name

    def joinTables(self,aliasname_left,aliasname_right,fake=False):
        aliasname_left_fa = self.findFather(aliasname_left)
        aliasname_right_fa = self.findFather(aliasname_right)
        self.aliasnames_fa[aliasname_left_fa] = self.total
        self.aliasnames_fa[aliasname_right_fa] = self.total
        self.left_son[self.total] = aliasname_left_fa
        self.right_son[self.total] = aliasname_right_fa
        self.aliasnames_root_set.add(self.total)

        self.left_aliasname[self.total] = aliasname_left
        self.right_aliasname[self.total] = aliasname_right
        if not fake:
            self.aliasnames_set[self.total] = self.aliasnames_set[aliasname_left_fa]|self.aliasnames_set[aliasname_right_fa]
            self.aliasnames_join_set[self.total] = (self.aliasnames_join_set[aliasname_left_fa]|self.aliasnames_join_set[aliasname_right_fa])-self.aliasnames_set[self.total]
            self.aliasnames_root_set.remove(aliasname_left_fa)
            self.aliasnames_root_set.remove(aliasname_right_fa)

        self.total += 1
    def recTable(self,node):
        if isinstance(node,int):
            res =  "("
            leftRes = self.recTable(self.left_son[node])
            if not self.left_son[node] in self.aliasnames:
                leftRes = leftRes[1:-1]

            res += leftRes + "\n"
            filter_list = []
            on_list = []
            if self.left_son[node] in self.filter_list:
                for condition in self.filter_list[self.left_son[node]]:
                    filter_list.append(str(condition))

            if self.right_son[node] in self.filter_list :
                for condition in self.filter_list[self.right_son[node]]:
                    filter_list.append(str(condition))

            cpList = []
            joined_aliasname = set([self.left_aliasname[node],self.right_aliasname[node]])
            # print('joined_aliasname',joined_aliasname)
            for left_table in self.aliasnames_set[self.left_son[node]]:
                for right_table,comparison in self.join_list[left_table]:
                    if right_table in self.aliasnames_set[self.right_son[node]]:
                        if (comparison.aliasname_list[1] in joined_aliasname and comparison.aliasname_list[0] in joined_aliasname):
                            cpList.append(str(comparison))
                        else:
                            on_list.append(str(comparison))
            if len(filter_list+on_list+cpList)>0:
                res += "inner join "
                res += self.recTable(self.right_son[node])
                res += "\non "
                res += " AND ".join(cpList + on_list+filter_list)
            else:
                res += "cross join "
                res += self.recTable(self.right_son[node])

            res += ")"
            return res
        else:
            return str(self.aliasname2fromtable[node])
    def encode_tree_regular(self,model, node_idx):

        def get_inputX(node):
            left_aliasname = self.left_aliasname[node]
            right_aliasname = self.right_aliasname[node]
            if not config.leafalias:
                left_emb =  model.leaf(torch.tensor([self.db_info.name2idx[self.aliasname2fullname[left_aliasname]]+40],device = device),self.table_fea_set[left_aliasname])
                right_emb = model.leaf(torch.tensor([self.db_info.name2idx[self.aliasname2fullname[right_aliasname]]+40],device = device),self.table_fea_set[right_aliasname])
            if config.leafalias:
                # print(self.aliasname2id[left_aliasname ]+40,self.aliasname2id[right_aliasname]+40)
                left_emb =  model.leaf(torch.tensor([self.aliasname2id[left_aliasname ]+40],device = device),self.table_fea_set[left_aliasname])
                right_emb = model.leaf(torch.tensor([self.aliasname2id[right_aliasname]+40],device = device),self.table_fea_set[right_aliasname])
            return model.inputX(left_emb[0],right_emb[0])
        def encode_node(node):
            if node in tlm:
                return tlm[node]
            if isinstance(node,int):
                left_h , left_c  = encode_node(self.left_son[node])
                right_h, right_c = encode_node(self.right_son[node])
                inputX = get_inputX(node)
                res =  model.childrenNode(left_h, left_c, right_h, right_c,inputX)
                if self.total > node + 1:
                    tlm[node] = res
            else:
                if not config.leafalias:
                    res =  model.leaf(torch.tensor([self.db_info.name2idx[self.aliasname2fullname[node]]],device = device),self.table_fea_set[node])
                else:
                    res =  model.leaf(torch.tensor([self.aliasname2id[node]],device = device),self.table_fea_set[node])
                tlm[node] = res
            return res
        encoding, _ = encode_node(node_idx)
        return encoding
    def encode_tree_fold(self,fold, node_idx):
        def get_inputX(node):
            left_aliasname = self.left_aliasname[node]
            right_aliasname = self.right_aliasname[node]
            if not config.leafalias:
                left_emb,c1 =  fold.add('leaf',self.db_info.name2idx[self.aliasname2fullname[left_aliasname]]+40,self.table_fea_set[left_aliasname]).split(2)
                right_emb,c2 = fold.add('leaf',self.db_info.name2idx[self.aliasname2fullname[right_aliasname]]+40,self.table_fea_set[right_aliasname]).split(2)
            else:
                left_emb,c1 =  fold.add('leaf',torch.tensor([self.aliasname2id[left_aliasname]+40],device = device),self.table_fea_set[left_aliasname]).split(2)
                right_emb,c2 = fold.add('leaf',torch.tensor([self.aliasname2id[right_aliasname]+40],device = device),self.table_fea_set[right_aliasname]).split(2)
            return fold.add('inputX',left_emb,right_emb)
        def encode_node(node):

            if isinstance(node,int):
                left_h, left_c = encode_node(self.left_son[node])
                right_h, right_c = encode_node(self.right_son[node])
                inputX = get_inputX(node)
                return fold.add('childrenNode',left_h, left_c, right_h, right_c,inputX).split(2)
            else:
                if not config.leafalias:
                    return fold.add('leaf',self.db_info.name2idx[self.aliasname2fullname[node]],self.table_fea_set[node]).split(2)
                else:
                    return fold.add('leaf',torch.tensor([self.aliasname2id[node]],device = device),self.table_fea_set[node]).split(2)
            return None
        encoding, _ = encode_node(node_idx)
        return encoding
    def hint(self,node):
        if isinstance(node,int):
            leftRes = self.hint(self.left_son[node])
            rightRes = self.hint(self.right_son[node])
            return '('+leftRes+' '+rightRes+')'
        else:
            # print(node)
            return node
    
    def toSql(self,):
        root = self.total - 1
        if config.use_hint:
            self.hintlist ="/*+\nLeading("+self.hint(root)+")\n*/\n"
            res = self.hintlist+self.sql
            # print(res)
            return res
        res = "select "+",\n".join([str(x) for x in self.target_table_list])+"\n"
        res  += "from " + self.recTable(root)[1:-1]
        res += ";"
        return res
    def plan2Cost(self):
        sql = self.toSql()
        return pgrunner.getLatency(self.sqlt,sql)
    def getResult(self,):
        sql = self.toSql()
        return pgrunner.getResult(self.sqlt,sql)

class sqlInfo:
    def __init__(self,pgrunner,sql,filename,trained = False):
        self.DPLantency = None
        self.DPCost = None
        self.bestLatency = None
        self.bestCost = None
        self.bestOrder = None
        self.plTime = None
        # pgRunner = pgRunner
        self.sql = sql
        self.filename = filename
        self.trained = trained
        self.useCost = False
        self.DPalready = False
        self.alias_cnt = 0
    def getDPlantecy(self,):
        if self.DPLantency == None:
            if not self.trained:
                self.DPLantency = pgrunner.getLatency(self,self.sql)
            else:
                self.DPLantency = config.maxTimeOut
                self.DPLantency = pgrunner.getLatency(self,self.sql)
                if self.DPLantency>=self.timeout():
                    self.useCost = True
                self.DPalready = True
        return self.DPLantency
    def getDPPlantime(self,):
        if self.plTime == None:
            self.plTime = pgrunner.getDPPlanTime(self,self.sql)
        return self.plTime
    def getDPCost(self,):
        if self.DPCost == None:
            self.DPCost = pgrunner.getCost(self,self.sql)
        return self.DPCost
    def timeout(self,):
        if self.trained and not self.DPalready:
            return config.maxTimeOut
        if self.DPLantency == None:
            if not self.trained:
                return 1000000
            else:
                return config.maxTimeOut
        # if self.getDPlantecy()>20*1000:
        #     return self.getDPlantecy()*4+self.getDPPlantime()
        if self.getDPlantecy()>10*1000:
            return self.getDPlantecy()*4+self.getDPPlantime()
        # if self.getDPlantecy()>2*1000:
        #     return self.getDPlantecy()*5+self.getDPPlantime()
        if self.getDPlantecy()>2*100:
            return self.getDPlantecy()*5+self.getDPPlantime()
        return 1000.0+self.getDPPlantime()
    def getBestOrder(self,):
        return self.bestOrder
    def updateBestOrder(self,latency,order):
        if self.bestOrder == None or self.bestLatency > latency:
            self.bestLatency = latency
            self.bestOrder = order





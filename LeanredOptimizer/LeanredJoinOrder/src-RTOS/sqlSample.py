
from JOBParser import TargetTable,FromTable,Comparison
max_column_in_table = 15
import torch
import torch
import torch.nn as nn
from itertools import count
import numpy as np

tree_lstm_memory = {}
class JoinTree:
    def __init__(self,sqlt,db_info,pgRunner,device):
        from psqlparse import parse_dict
        global tree_lstm_memory
        tree_lstm_memory  ={}
        self.sqlt = sqlt
        self.sql = self.sqlt.sql
        parse_result = parse_dict(self.sql)[0]["SelectStmt"]
        self.target_table_list = [TargetTable(x["ResTarget"]) for x in parse_result["targetList"]]
        self.from_table_list = [FromTable(x["RangeVar"]) for x in parse_result["fromClause"]]
        self.aliasname2fullname = {}
        self.pgrunner = pgRunner
        self.device = device
        self.aliasname2fromtable={}
        for table in self.from_table_list:
            self.aliasname2fromtable[table.getAliasName()] = table
            self.aliasname2fullname[table.getAliasName()] = table.getFullName()
        self.aliasnames = set(self.aliasname2fromtable.keys())
        self.comparison_list =[Comparison(x) for x in parse_result["whereClause"]["BoolExpr"]["args"]]
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
            self.table_fea_set[aliasname] = [0.0]*max_column_in_table*2

        ##提取所有的Join和filter
        self.join_candidate = set()
        self.join_matrix=[]
        for aliasname in self.aliasnames_root_set:
            self.join_list[aliasname] = []
        for idx in range(len(self.db_info)):
            self.join_matrix.append([0]*len(self.db_info))
        for comparison in self.comparison_list:
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
                idx0 = self.db_info.name2idx[left_fullname]
                idx1 = self.db_info.name2idx[right_fullname]
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
                self.table_fea_set[left_aliasname][table_idx * 2 + 1] += self.pgrunner.getSelectivity(str(self.aliasname2fromtable[comparison.aliasname_list[0]]),str(comparison))


        for aliasname in self.aliasnames_root_set:
            self.table_fea_set[aliasname] = torch.tensor(self.table_fea_set[aliasname],device = self.device).reshape(1,-1).detach()
            self.aliasnames_set[aliasname] = set([aliasname])
            for y in self.join_list[aliasname]:
                if aliasname not in self.aliasnames_join_set:
                    self.aliasnames_join_set[aliasname] = set()
                self.aliasnames_join_set[aliasname].add(y[0])


        predice_list_dict={}
        for table in self.db_info.tables:
            predice_list_dict[table.name] = [0] * len(table.column2idx)
        for filter_table in self.filter_list:
            for comparison in self.filter_list[filter_table]:
                aliasname = comparison.aliasname_list[0]
                fullname = self.aliasname2fullname[aliasname]
                table = self.db_info.name2table[fullname]
                for column in comparison.column_list:
                    columnidx = table.column2idx[column]
                    predice_list_dict[self.aliasname2fullname[filter_table]][columnidx] = 1
        self.predice_feature = []
        for fullname in predice_list_dict:
            self.predice_feature+= predice_list_dict[fullname]
        self.predice_feature = np.asarray(self.predice_feature).reshape(1,-1)
        self.join_matrix = torch.tensor(np.asarray(self.join_matrix).reshape(1,-1),device = self.device,dtype = torch.float32)

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
            left_emb =  model.leaf(torch.tensor([self.db_info.name2idx[self.aliasname2fullname[left_aliasname]]+25],device = self.device),self.table_fea_set[left_aliasname])
            right_emb = model.leaf(torch.tensor([self.db_info.name2idx[self.aliasname2fullname[right_aliasname]]+25],device = self.device),self.table_fea_set[right_aliasname])
            return model.inputX(left_emb[0],right_emb[0])
        def encode_node(node):
            if node in tree_lstm_memory:
                return tree_lstm_memory[node]
            if isinstance(node,int):
                left_h, left_c = encode_node(self.left_son[node])
                right_h, right_c = encode_node(self.right_son[node])
                inputX = get_inputX(node)
                res =  model.childrenNode(left_h, left_c, right_h, right_c,inputX)
                if self.total > node + 1:
                    tree_lstm_memory[node] = res
            else:
                res =  model.leaf(torch.tensor([self.db_info.name2idx[self.aliasname2fullname[node]]],device = self.device),self.table_fea_set[node])
                tree_lstm_memory[node] = res

            return res
        encoding, _ = encode_node(node_idx)
        return encoding
    def encode_tree_fold(self,fold, node_idx):
        def get_inputX(node):
            left_aliasname = self.left_aliasname[node]
            right_aliasname = self.right_aliasname[node]
            left_emb,c1 =  fold.add('leaf',self.db_info.name2idx[self.aliasname2fullname[left_aliasname]]+25,self.table_fea_set[left_aliasname]).split(2)
            right_emb,c2 = fold.add('leaf',self.db_info.name2idx[self.aliasname2fullname[right_aliasname]]+25,self.table_fea_set[right_aliasname]).split(2)
            return fold.add('inputX',left_emb,right_emb)
        def encode_node(node):

            if isinstance(node,int):
                left_h, left_c = encode_node(self.left_son[node])
                right_h, right_c = encode_node(self.right_son[node])
                inputX = get_inputX(node)
                return fold.add('childrenNode',left_h, left_c, right_h, right_c,inputX).split(2)
            else:
                return fold.add('leaf',self.db_info.name2idx[self.aliasname2fullname[node]],self.table_fea_set[node]).split(2)
            return None
        encoding, _ = encode_node(node_idx)
        return encoding
    def toSql(self,):
        root = self.total - 1
        res = "select "+",\n".join([str(x) for x in self.target_table_list])+"\n"
        res  += "from " + self.recTable(root)[1:-1]
        res += ";"
        return res
    def plan2Cost(self):
        sql = self.toSql()
        return self.pgrunner.getLatency(self.sqlt,sql)

class sqlInfo:
    def __init__(self,pgRunner,sql,filename):
        self.DPLantency = None
        self.DPCost = None
        self.bestLatency = None
        self.bestCost = None
        self.bestOrder = None
        self.plTime = None
        self.pgRunner = pgRunner
        self.sql = sql
        self.filename = filename
    def getDPlantecy(self,):
        if self.DPLantency == None:
            self.DPLantency = self.pgRunner.getLatency(self,self.sql)
        return self.DPLantency
    def getDPPlantime(self,):
        if self.plTime == None:
            self.plTime = self.pgRunner.getDPPlanTime(self,self.sql)
        return self.plTime
    def getDPCost(self,):
        if self.DPCost == None:
            self.DPCost = self.pgRunner.getCost(self,self.sql)
        return self.DPCost
    def timeout(self,):
        if self.DPLantency == None:
            return 1000000
        return self.getDPlantecy()*4+self.getDPPlantime()
    def getBestOrder(self,):
        return self.bestOrder
    def updateBestOrder(self,latency,order):
        if self.bestOrder == None or self.bestLatency > latency:
            self.bestLatency = latency
            self.bestOrder = order






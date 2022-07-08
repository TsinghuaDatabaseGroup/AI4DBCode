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
import torch
from torch.nn import init
import torchfold
import torch.nn as nn
from ImportantConfig import Config

config = Config()

class TreeLSTM(nn.Module):
    def __init__(self, num_units):
        super(TreeLSTM, self).__init__()
        self.num_units = num_units
        self.FC1 = nn.Linear(num_units, 5 * num_units)
        self.FC2 = nn.Linear(num_units, 5 * num_units)
        self.FC0 = nn.Linear(num_units, 5 * num_units)
        self.LNh = nn.LayerNorm(num_units,elementwise_affine = False)
        self.LNc = nn.LayerNorm(num_units,elementwise_affine = False)
    def forward(self, left_in, right_in,inputX):
        lstm_in = self.FC1(left_in[0])
        lstm_in += self.FC2(right_in[0])
        lstm_in += self.FC0(inputX)
        a, i, f1, f2, o = lstm_in.chunk(5, 1)
        c = (a.tanh() * i.sigmoid() + f1.sigmoid() * left_in[1] +
             f2.sigmoid() * right_in[1])
        h = o.sigmoid() * c.tanh()
        h = self.LNh(h)
        return h,c
class TreeRoot(nn.Module):
    def __init__(self,num_units):
        super(TreeRoot, self).__init__()
        self.num_units = num_units
        self.FC = nn.Linear(num_units, num_units)
        if config.rootPool == 'meanPool':
            self.sum_pooling = nn.AdaptiveAvgPool2d((1,num_units))
        else:
            self.sum_pooling = nn.AdaptiveMaxPool2d((1,num_units))
        
        # self.sum_pooling = nn.AdaptiveMaxPool2d((1,num_units))
        # self.max_pooling = nn.AdaptiveAvgPool2d((1,num_units))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, tree_list):

        return self.relu(self.FC(self.sum_pooling(tree_list)).view(-1,self.num_units))

class SPINN(nn.Module):

    def __init__(self, n_classes, size, n_words, mask_size,device,max_column_in_table = 15):
        super(SPINN, self).__init__()
        self.size = size
        self.tree_lstm = TreeLSTM(size)
        self.tree_root = TreeRoot(size)
        self.FC = nn.Linear(size*2, size)
        self.table_embeddings = nn.Embedding(n_words, size)#2 * max_column_in_table * size)
        self.column_embeddings = nn.Embedding(n_words, (1+2 * max_column_in_table) * size)
        self.out = nn.Linear(size*2, size)
        self.out2 = nn.Linear(size, n_classes)
        self.outFc = nn.Linear(mask_size, size)

        if config.rootPool == 'meanPool':
            self.max_pooling = nn.AdaptiveAvgPool2d((1,size))
        else:
            self.max_pooling = nn.AdaptiveMaxPool2d((1,size))
        self.max_pooling = nn.AdaptiveMaxPool2d((1,size))
        self.relu = nn.ReLU()
        self.sigmoid = nn.ReLU()
        self.leafFC = nn.Linear(size, size)
        self.sigmoid = nn.Sigmoid()
        self.LN1 = nn.LayerNorm(size,)
        self.LN2 = nn.LayerNorm(size,)
        self.max_column_in_table = max_column_in_table
        self.leafLn = nn.LayerNorm(size,elementwise_affine = False)
        self.device = device
        self.sigmoid = nn.Sigmoid()

    def leaf(self, word_id, table_fea=None):
        # print('tlstm_wi',word_id)
        all_columns = table_fea.view(-1,self.max_column_in_table*2+1,1)*self.column_embeddings(word_id).reshape(-1,2 * self.max_column_in_table+1,self.size)
        all_columns = self.relu(self.leafFC(all_columns))
        table_emb = self.max_pooling(all_columns.view(-1,self.max_column_in_table*2+1,self.size)).view(-1,self.size)
        return self.leafLn(table_emb), torch.zeros(word_id.size()[0], self.size,device = self.device,dtype = torch.float32)
    def inputX(self,left_emb,right_emb):
        cat_emb = torch.cat([left_emb,right_emb],dim = 1)
        return self.relu(self.FC(cat_emb))
    def childrenNode(self, left_h, left_c, right_h, right_c,inputX):
        return self.tree_lstm((left_h, left_c), (right_h, right_c),inputX)
    def root(self,tree_list):
        return self.tree_root(tree_list).view(-1,self.size)
    def logits(self, encoding,join_matrix,prt=False):
        encoding = self.root(encoding.view(1,-1,self.size))
        # if prt:
        #     print(encoding)
        matrix = self.relu(self.outFc(join_matrix))
        # outencoding = torch.cat([encoding,encoding],dim = 1)
        outencoding = torch.cat([encoding,matrix],dim = 1)
        return self.out2(self.relu(self.out(outencoding)))
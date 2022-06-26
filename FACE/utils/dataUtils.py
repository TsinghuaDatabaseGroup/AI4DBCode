#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os


# # Help Functions

# ## LoadTable

# In[2]:


COLS = {
    'power': ['c0', 'c1', 'c2', 'c3', 'c4', 'c5'],
    'BJAQ' : ['PM2.5', 'PM10', 'NO2', 'O3', 'TEMP'],
    
}


# In[3]:


def LoadTable(dataset_name):
    data_PATH = '/home/jiayi/disk/gits/Jupyters/FACE-June-Release/data/'
    path = os.path.join(data_PATH, '{}.npy'.format(dataset_name))
    data = np.load(path).astype(np.float32)
    cols = COLS[dataset_name]

    print('data shape:', data.shape)
    n, dim = data.shape
    
    return pd.DataFrame(data, columns=cols), n, dim


# In[ ]:





# ## completeColumns

# In[4]:


def completeColumns(table, columns, operators, vals):
    """ complete columns not used in query"""
    ncols = table.dim
    cs = table.columns
    os, vs = [None] * ncols, [None] * ncols

    for c, o, v in zip(columns, operators, vals):
        idx = table.getColID(c if isinstance(c, str) else c.name)
        os[idx] = o
        vs[idx] = v

    return cs, os, vs


# In[ ]:





# In[5]:


def FillInUnqueriedColumns(table, columns, operators, vals):
    ncols = len(table.columns)
    cs = table.columns
    os, vs = [None] * ncols, [None] * ncols

    for c, o, v in zip(columns, operators, vals):
        assert isinstance(c, str)
        idx = table.ColumnIndex(c)
        os[idx] = o
        vs[idx] = v

    return cs, os, vs


# ## LoadOracleCardinalities

# In[6]:


def LoadOracleCardinalities(dataset_name, querySeed=1234):
    
    ORACLE_CARD_FILES = {
        'power': '/home/jiayi/flow/datasets/oracles/{}_rng-{}.csv'.format(dataset_name, querySeed),
        'BJAQ': '/home/jiayi/flow/datasets/oracles/{}_rng-{}.csv'.format(dataset_name, querySeed),
    }

    path = ORACLE_CARD_FILES.get(dataset_name, None)
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        print('Found oracle card!')
        return df.values.reshape(-1)
    print('Can not find oracle card! at')
    print(path)
    return None


# In[ ]:





# # DataWrapper

# In[7]:



filterNum = {
    'power': (3,6),
    'BJAQ' : (2, 4),
}

sensible = {
    'power' : np.ones(6),
    'BJAQ'  : np.ones(5),
}


deltas = {
    'power': np.zeros(6),
    'BJAQ' : np.array([1, 1, 1, 1, 0.1])
}


Norm_us = {
    'BJAQ' : np.array([79.9326, 105.07354, 51.070656, 57.876205, 13.568575]),
}
Norm_ss = {
    'BJAQ' : np.array([80.15541, 91.38018, 35.06305, 56.71038, 11.425453]),
}


# In[8]:


OPS = {
    '>':np.greater,
    '<':np.less,
    '>=':np.greater_equal,
    '<=':np.less_equal,
    '=':np.equal,
}


# In[9]:


class DataWrapper():
    def __init__(self, data, dataset_name):
        
        self.data = data
        self.n = data.shape[0]
        self.cardinality = data.shape[0]
        self.dim = data.shape[1]
        self.columns = data.columns
        self.dataset_name = dataset_name
        
        self.Mins = data.min(axis=0)
        self.Maxs = data.max(axis=0)
        
        self.minFilter, self.maxFilter = filterNum[self.dataset_name]
        self.sensible_to_do_range = sensible[self.dataset_name]
        self.colMap = {col:i for i,col in enumerate(self.columns)}
        self.delta = deltas[self.dataset_name]

    
    def getColID(self, col):
        return self.colMap[col]
    
    def getCateColumns(self, cols):
        cols = [self.getColID(col) for col in cols]
        return self.sensible_to_do_range[cols]
        
    
    
    def GetUnNormalizedValue(self, col_id, val):
        if self.dataset_name=='power':
            return val
        U = Norm_us[self.dataset_name]
        S = Norm_ss[self.dataset_name]
        ret = (val - U[col_id])/S[col_id]
        return ret



    def GetLegalRange(self, col, op, val):
        """ legal range for a column """
        col_id = self.getColID(col)        
        add_one = self.delta[col_id]
        if op == '=':
            l = self.GetUnNormalizedValue(col_id, val)
            r = self.GetUnNormalizedValue(col_id, val + add_one)
        elif op == '>' or op =='>=':
            l = self.GetUnNormalizedValue(col_id, val)
            r = self.GetUnNormalizedValue(col_id, self.Maxs[col_id] + add_one)
        elif op == '<' or op =='<=':
            l = self.GetUnNormalizedValue(col_id, self.Mins[col_id])
            r = self.GetUnNormalizedValue(col_id, val + add_one)
        elif op == 'in':
            l = self.GetUnNormalizedValue(col_id, val[0])
            r = self.GetUnNormalizedValue(col_id, val[1] + add_one)

        return [l,r]
    
    
    
    def getLegalRangeQuery(self, query):
        """legal range for a query"""
        cols, ops, vals = query
        cols, ops, vals = completeColumns(self, cols, ops, vals)
        

        legal_list = [[0., 1.]] * len(vals)
        i = 0
        for co, op, val_i in zip(cols, ops, vals):
            col_id = self.getColID(co)
            if val_i is None:
                legal_list[i] = self.GetLegalRange(co, 'in', [self.Mins[col_id], self.Maxs[col_id]])
            else:
                legal_list[i] = self.GetLegalRange(co, op, val_i)
            i = i + 1
        return legal_list

    
    def getLegalRangeNQuery(self, queries):
        """ legal ranges for N queries"""
        legal_lists = []

        for query in queries:
            legal_lists.append(self.getLegalRangeQuery(query))
        return legal_lists

    
    def generateQuery(self, rng):
        """ generate a query """

        num_filters = rng.randint(self.minFilter, self.maxFilter)

        loc = rng.randint(0, self.cardinality)
        tuple0 = self.data.iloc[loc]
        tuple0 = tuple0.values
        loc = rng.randint(0, self.cardinality)
        tuple1 = self.data.iloc[loc]
        tuple1 = tuple1.values

        idxs = rng.choice(len(self.columns), replace=False, size=num_filters)
        cols = np.take(self.columns, idxs)

        ops = rng.choice(['<=', '>='], size=num_filters)
        ops_all_eqs = ['='] * num_filters
        

        sensible_to_do_range = self.getCateColumns(cols)
        ops = np.where(sensible_to_do_range, ops, ops_all_eqs)
        vals = tuple0[idxs]
        vals = list(vals)

        if self.dataset_name == 'BJAQ':
            ops = ['in'] * len(ops)

        tuple0 = tuple0[idxs]
        tuple1 = tuple1[idxs]
        for i, op in enumerate(ops):
            if op == 'in':
                vals[i] = ([tuple0[i], tuple1[i]] if tuple0[i]<=tuple1[i] else [tuple1[i], tuple0[i]])

        return cols, ops, vals

    
    def generateNQuery(self, n, rng):
        """ generate N queries """
        ret = []
        for i in range(n):
            ret.append(self.generateQuery(rng))
        return ret
    
    def getOracle(self, query):
        """ get oracle result for a query """
        columns, operators, vals = query
        assert len(columns) == len(operators) == len(vals)

        bools = None
        for c, o, v in zip(columns, operators, vals):
            c = self.data[c]
            if o in OPS.keys():
                inds = OPS[o](c, v)
            else:
                if o == 'in' or o == 'IN':   
                    inds = np.greater_equal(c, v[0])
                    inds &= np.less_equal(c, v[1])

            if bools is None:
                bools = inds
            else:
                bools &= inds
        c = bools.sum()
        return c

    def getAndSaveOracle(self, queries, querySeed=2345):
        """ Calculate oracle results for input queries and save the results"""
        n = len(queries)
        oracle_cards = np.empty(2000)
        for i, query in enumerate(queries):
            oracle_cards[i] = self.getOracle(query)
        oracle_cards = oracle_cards.astype(np.int)
        df = pd.DataFrame(oracle_cards, columns=['true_card'])
        
        """ Change it to your own path """
        print("Save oracle results to :")
        print('/home/jiayi/flow/datasets/oracles/{}_rng-{}.csv'.format(self.dataset_name, querySeed))
        df.to_csv('/home/jiayi/flow/datasets/oracles/{}_rng-{}.csv'.format(self.dataset_name, querySeed),
                 index=False)
        return


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





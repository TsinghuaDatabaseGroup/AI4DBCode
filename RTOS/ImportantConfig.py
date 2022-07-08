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
class Config:
    def __init__(self,):
        self.sytheticDir = "join-order-benchmark/"
        self.JOBDir = "join-order-benchmark/"
        self.rootPool = "meanPool"
        self.schemaFile = "schema.sql"
        self.dbName = "imdbload"
        self.userName = ""
        self.password = ""
        self.usegpu = True
        self.ip = "127.0.0.1"
        self.port = 5432
        self.use_hint = True
        self.maxTimeOut = 10*1000
        self.batchsize = 16
        self.gen_time = 200
        self.gpu_device = 0
        self.EPS_START = 0.8
        self.EPS_END = 0.2
        self.EPS_DECAY = 30*10
        self.memory_size = 10000
        self.learning_rate = 10e-3
        self.maxR = 4
        self.baselineValue = 1.4
        self.isCostTraining = True
        self.latencyRecord = True
        self.leafalias  = True
        self.latencyRecordFile = 'l_t.json'
        self.max_parallel_workers_per_gather = 1
        self.max_parallel_workers = 1
        self.enable_mergejoin = True
        self.enable_hashjoin = True
        

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
import psycopg2
import json
from math import log
from ImportantConfig import Config
class PGConfig:
    def __init__(self):
        self.keepExecutedPlan =True
        self.maxTimes = 5
        self.maxTime = 300000

LatencyDict = {}
selectivityDict = {}
LatencyRecordFileHandle = None
config  = Config()

class PGGRunner:
    def __init__(self,dbname = '',user = '',password = '',host = '',port = '',isCostTraining = True,latencyRecord = True,latencyRecordFile = "RecordFile.json"):
        """
        :param dbname:
        :param user:
        :param password:
        :param host:
        :param port:
        :param latencyRecord:-1:loadFromFile
        :param latencyRecordFile:
        """
        self.con = psycopg2.connect(database=dbname, user=user,
                               password=password, host=host, port=port)
        self.cur = self.con.cursor()
        self.config = PGConfig()
        self.isLatencyRecord = latencyRecord
        global LatencyRecordFileHandle
        self.isCostTraining = isCostTraining
        if config.enable_mergejoin:
            self.cur.execute("set enable_mergejoin = true")
        else:
            self.cur.execute("set enable_mergejoin = false")
        if config.enable_hashjoin:
            self.cur.execute("set enable_hashjoin = true")
        else:
            self.cur.execute("set enable_hashjoin = false")
        # self.cur.execute("set enable_hashjoin = false")
        
        if latencyRecord:
            LatencyRecordFileHandle = self.generateLatencyPool(latencyRecordFile)


    def generateLatencyPool(self,fileName):
        """
        :param fileName:
        :return:
        """
        import os
        import json
        if os.path.exists(fileName):
            f = open(fileName,"r")
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                global LatencyDict
                LatencyDict[data[0]] = data[1]
            f = open(fileName,"a")
        else:
            f = open(fileName,"w")
        return f
    def getLatency(self, sql,sqlwithplan):
        """
        :param sql:a sqlSample object.
        
        :return: the latency of sql
        """
        if self.isCostTraining:
            return self.getCost(sql,sqlwithplan)
        if sql.useCost:
            return self.getCost(sql,sqlwithplan)
        global LatencyDict
        if self.isLatencyRecord:
            if sqlwithplan in LatencyDict:
                return LatencyDict[sqlwithplan]
        thisQueryCost = self.getCost(sql,sqlwithplan)
        if thisQueryCost / sql.getDPCost()<1000000000:
            try:
                
                self.cur.execute("SET statement_timeout = "+str(int(sql.timeout()))+ ";")
                self.cur.execute("set max_parallel_workers = "+str(config.max_parallel_workers)+";")
                self.cur.execute("set max_parallel_workers_per_gather = "+str(config.max_parallel_workers_per_gather)+";")
                
                if config.use_hint and sqlwithplan.find("/*")>-1:
                    self.cur.execute("set join_collapse_limit = 20;")
                    self.cur.execute("set geqo_threshold = 20;")
                else:
                    self.cur.execute("set join_collapse_limit = 1;")
                    if not sql.trained:
                        self.cur.execute("set geqo_threshold = 20;")
                    else:
                        self.cur.execute("set geqo_threshold = 2;")
                self.cur.execute("explain (COSTS, FORMAT JSON, ANALYSE) "+sqlwithplan)
                rows = self.cur.fetchall()
                row = rows[0][0]
                import json
                # print(json.dumps(rows[0][0][0]['Plan']))
                afterCost = rows[0][0][0]['Plan']['Actual Total Time']
                # print(1)
            except:
                self.con.commit()
                afterCost = max(thisQueryCost / sql.getDPCost()*sql.getDPlantecy(),sql.timeout())
            # print("PGUtils.py excited!!!",afterCost)
        else:
            afterCost = max(thisQueryCost / sql.getDPCost()*sql.getDPlantecy(),sql.timeout())
        afterCost += 5
        if self.isLatencyRecord:
            LatencyDict[sqlwithplan] =  afterCost
            global LatencyRecordFileHandle
            import json
            LatencyRecordFileHandle.write(json.dumps([sqlwithplan,afterCost])+"\n")
            LatencyRecordFileHandle.flush()
        return afterCost
    def getResult(self, sql,sqlwithplan):
        """
        :param sql:a sqlSample object
        :return: the latency of sql
        """
        self.cur.execute("SET statement_timeout = 600000;")
        # self.cur.execute("set max_parallel_workers = "+str(config.max_parallel_workers)+";")
        # self.cur.execute("set max_parallel_workers_per_gather = "+str(config.max_parallel_workers_per_gather)+";")
        
        if config.use_hint and sqlwithplan.find("/*")>-1:
            self.cur.execute("set join_collapse_limit = 20;")
            self.cur.execute("set geqo_threshold = 20;")
        else:
            self.cur.execute("set join_collapse_limit = 1;")
            self.cur.execute("set geqo_threshold = 2;")
        # self.cur.execute("SET statement_timeout =  4000;")
        import time
        st = time.time()
        self.cur.execute(sqlwithplan)
        rows = self.cur.fetchall()
        et = time.time()
        print('runtime : ',et-st)
        return rows
    def getCost(self,sql,sqlwithplan):
        """
        :param sql: a sqlSample object
        :return: the cost of sql
        """
        self.cur.execute("set max_parallel_workers = "+str(config.max_parallel_workers)+";")
        self.cur.execute("set max_parallel_workers_per_gather = "+str(config.max_parallel_workers_per_gather)+";")
        
        if config.use_hint and sqlwithplan.find("/*")>-1:
            self.cur.execute("set join_collapse_limit = 222;")
            self.cur.execute("set geqo_threshold = 202;")
        else:
            self.cur.execute("set join_collapse_limit = 1;")
            self.cur.execute("set geqo_threshold = 2;")
        self.cur.execute("SET statement_timeout =  40000;")
        self.cur.execute("EXPLAIN "+sqlwithplan)
        rows = self.cur.fetchall()
        row = rows[0][0]
        afterCost = float(rows[0][0].split("cost=")[1].split("..")[1].split(" ")[
                              0])
        self.con.commit()
        return afterCost
    
    def getPlan(self,sqlwithplan):
        """
        :param sql: a sqlSample object
        :return: the cost of sql
        """
        self.cur.execute("set max_parallel_workers = "+str(config.max_parallel_workers)+";")
        self.cur.execute("set max_parallel_workers_per_gather = "+str(config.max_parallel_workers_per_gather)+";")
        
        if config.use_hint and sqlwithplan.find("/*")>-1:
            self.cur.execute("set join_collapse_limit = 20;")
            self.cur.execute("set geqo_threshold = 20;")
        else:
            self.cur.execute("set join_collapse_limit = 1;")
            self.cur.execute("set geqo_threshold = 2;")
        self.cur.execute("SET statement_timeout =  4000;")
        sqlwithplan = sqlwithplan +";"
        self.cur.execute("EXPLAIN (COSTS, FORMAT JSON) "+sqlwithplan)
        rows = self.cur.fetchall()
        import json
        return rows

    def getDPPlanTime(self,sql,sqlwithplan):
        """
        :param sql: a sqlSample object
        :return: the planTime of sql
        """
        import time
        startTime = time.time()
        cost = self.getCost(sql,sqlwithplan)
        plTime = time.time()-startTime
        return plTime
    def getSelectivity(self,table,whereCondition):
        global selectivityDict
        if whereCondition in selectivityDict:
            return selectivityDict[whereCondition]
        # if config.isCostTraining:
        self.cur.execute("SET statement_timeout = "+str(int(100000))+ ";")
        totalQuery = "select * from "+table+";"
        #     print(totalQuery)

        self.cur.execute("EXPLAIN "+totalQuery)
        rows = self.cur.fetchall()[0][0]
        #     print(rows)
        #     print(rows)
        total_rows = int(rows.split("rows=")[-1].split(" ")[0])

        resQuery = "select * from "+table+" Where "+whereCondition+";"
        # print(resQuery)
        self.cur.execute("EXPLAIN  "+resQuery)
        rows = self.cur.fetchall()[0][0]
        #     print(rows)
        select_rows = int(rows.split("rows=")[-1].split(" ")[0])
        selectivityDict[whereCondition] = -log(select_rows/total_rows)
        # else:
        #     self.cur.execute("SET statement_timeout = "+str(int(100000))+ ";")
        #     totalQuery = "select count(*) from "+table+";"
        #     self.cur.execute(totalQuery)
        #     total_rows = self.cur.fetchall()[0][0]

        #     resQuery = "select count(*) from "+table+" Where "+whereCondition+";"
            
        #     self.cur.execute(resQuery)
        #     select_rows = self.cur.fetchall()[0][0]+1
        #     selectivityDict[whereCondition] = -log(select_rows/total_rows)
        return selectivityDict[whereCondition]
latencyRecordFile = config.latencyRecordFile
from itertools import count
from pathlib import Path

pgrunner = PGGRunner(config.dbName,config.userName,config.password,config.ip,config.port,isCostTraining=config.isCostTraining,latencyRecord = config.latencyRecord,latencyRecordFile = latencyRecordFile)
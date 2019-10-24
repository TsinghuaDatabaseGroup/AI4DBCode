# -*- coding: utf-8 -*-
"""
desciption: action for database
"""
import MySQLdb
from base import Err,cdb_logger,os_quit



class database:
#注，python的self等于其它语言的this
    def __init__(self, dbhost=None, dbport=None, dbuser=None, dbpwd=None, dbname=None):    
        self._dbname = dbname   
        self._dbhost = dbhost 
        self._dbuser = dbuser
        self._dbpassword = dbpwd
        self._dbport = dbport
        self._logger = cdb_logger

        self._conn = self.connectMySQL()
        if(self._conn):
            self._cursor = self._conn.cursor()


    #数据库连接
    def connectMySQL(self):
        conn = False
        try:
            conn = MySQLdb.connect(host=self._dbhost,
                    user=self._dbuser,
                    passwd=self._dbpassword,
                    db=self._dbname,
                    port=self._dbport,
                    )
        except Exception,data:
            self._logger.error("connect database failed, %s" % data)
            os_quit(Err.MYSQL_CONNECT_ERR,"host:%s,port:%s,user:%s" % (self._dbhost,self._dbport,self._dbuser))
            conn = False
        return conn


    #获取查询结果集
    def fetch_all(self, sql , json=True):
        res = ''
        if(self._conn):
            try:
                self._cursor.execute(sql)
                res = self._cursor.fetchall()
                if json :
                    columns = [col[0] for col in self._cursor.description]
                    return [
                        dict(zip(columns, row))
                        for row in res
                    ]
            except Exception, data:
                res = False
                self._logger.warn("query database exception, %s" % data)
                os_quit(Err.MYSQL_EXEC_ERR,sql)
        return res


    def update(self, sql):
        flag = False
        if(self._conn):
            try:
                self._cursor.execute(sql)
                self._conn.commit()
                flag = True
            except Exception, data:
                flag = False
                self._logger.warn("update database exception, %s" % data)
                os_quit(Err.MYSQL_EXEC_ERR,sql)
        return flag

    #关闭数据库连接
    def close(self):
        if(self._conn):
            try:
                if(type(self._cursor)=='object'):
                    self._cursor.close()
                if(type(self._conn)=='object'):
                    self._conn.close()
            except Exception, data:
                self._logger.warn("close database exception, %s,%s,%s" % (data, type(self._cursor), type(self._conn)))
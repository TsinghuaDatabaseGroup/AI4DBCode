# -*- coding: utf-8 -*-
"""
desciption: err or other constant information
"""
from enum import IntEnum
import logging
import os
import requests



def Logger(name="default_log", logger_level="debug" ):
	'''
	logger  返回一个logger对象
	IN : name logger的名字
		 logger_level 日志的等级 
	OUT : logger
	''' 
	logname = name.split("/")[-1][:-3]
	logger_levels = {
		'debug':logging.DEBUG,
		'info':logging.INFO,
		'warning':logging.WARNING,
		'error':logging.ERROR,
		'crit':logging.CRITICAL
	}#日志级别关系映射

	logger = logging.getLogger(logname)
	logger.setLevel(logger_levels.get(logger_level))
	
	return logger

cdb_logger = Logger("init")

def init_logger(task_id,write_console = False,write_file=True):
    global cdb_logger
    fmt = "[%(asctime)s][%(levelname)s][%(process)d-%(filename)s:%(lineno)d] : %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    if write_file:
        fh = logging.FileHandler(CONST.FILE_LOG % task_id,mode='a')#
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        cdb_logger.addHandler(fh)
    if write_console:
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(logging.Formatter(fmt, datefmt))
        cdb_logger.addHandler(sh)

class CONST:
    TASK_ID = -1


    PROJECT_DIR = "/usr/local/cdbtune/"
    LOG_PATH = PROJECT_DIR + "log/"
    SCRIPT_PATH = PROJECT_DIR + "scripts/"
    LOG_SYSBENCH_PATH = LOG_PATH+ "sysbench/"

    FILE_LOG = LOG_PATH + "%d.log"
    FILE_LOG_SYSBENCH = LOG_SYSBENCH_PATH + "%d_%s.log"
    FILE_LOG_BEST = LOG_PATH + "%d_bestnow.log"

    BASH_SYSBENCH = SCRIPT_PATH + "run_sysbench.sh"
    BASH_TPCC = SCRIPT_PATH + "run_tpcc.sh"

    cdb_public_api = "http://%s/cdb2/fun_logic/cgi-bin/public_api"
    URL_SET_PARAM = cdb_public_api +"/set_mysql_param.cgi"
    URL_QUERY_SET_PARAM = cdb_public_api +"/query_set_mysql_param_task.cgi"


    cdbtune_server = "http://127.0.0.1:9119"
    URL_INSERT_RESULT = cdbtune_server +"/insert_task_result"
    URL_UPDATE_TASK =  cdbtune_server +"/update_task"



class Err(IntEnum):
    INPUT_ERROR = 101
    HTTP_REQUERT_ERR = 103

    RUN_SYSYBENCH_FAILED = 201
    SET_MYSQL_PARAM_FAILED = 202

    MYSQL_CONNECT_ERR = 301
    MYSQL_EXEC_ERR = 302


class Err_Detail:
    Desc = dict()
    @classmethod 
    def add_desc(self,err,desc):
        self.Desc[err]=desc

Err_Detail.add_desc(Err.INPUT_ERROR,"输入错误")
Err_Detail.add_desc(Err.HTTP_REQUERT_ERR,"链接请求错误")

Err_Detail.add_desc(Err.RUN_SYSYBENCH_FAILED,"sysbench 压测出现异常")
Err_Detail.add_desc(Err.SET_MYSQL_PARAM_FAILED,"mysql系统参数设置失败")

Err_Detail.add_desc(Err.MYSQL_CONNECT_ERR,"mysql 链接错误")
Err_Detail.add_desc(Err.MYSQL_EXEC_ERR,"输入错误")

def os_quit(err,detail=""):
    err_str = "err: %d, %s, %s" % (err , Err_Detail.Desc[err] , detail)
    data = dict()
    data["task_id"] = CONST.TASK_ID
    data["error"] = err_str
    cdb_logger.error(err_str)
    cdb_logger.info("update task status %s", requests.post(CONST.URL_UPDATE_TASK, data))
    os._exit(int(err))


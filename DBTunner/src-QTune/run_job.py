import threading
import time
import queue
import pymysql
from configs import parse_args
from dbutils.pooled_db import PooledDB

args = parse_args()
lock = threading.Lock()
total_lat = 0
error_query_num = 0

POOL = None


# 把任务放入队列中
class Producer(threading.Thread):
    def __init__(self, name, queue, workload):
        self.__name = name
        self.__queue = queue
        self.workload = workload
        super(Producer, self).__init__()

    def run(self):
        for index, query in enumerate(self.workload):
            self.__queue.put(str(index) + "~#~" + query)


# 线程处理任务
class Consumer(threading.Thread):
    def __init__(self, name, queue):
        self.__name = name
        self.__queue = queue
        super(Consumer, self).__init__()

    def run(self):
        while not self.__queue.empty():
            query = self.__queue.get()
            try:
                consumer_process(query)
            finally:
                self.__queue.task_done()


def consumer_process(task_key):
    query = task_key.split('~#~')[1]
    if query:

        start = time.time()
        result = mysql_query(query)
        end = time.time()
        interval = end - start

        if result:
            lock.acquire()
            global total_lat
            total_lat += interval
            lock.release()

        else:
            global error_query_num
            lock.acquire()
            error_query_num += 1
            lock.release()


def startConsumer(thread_num, queue):
    t_consumer = []
    for i in range(thread_num):
        c = Consumer(i, queue)
        c.setDaemon(True)
        c.start()
        t_consumer.append(c)
    return t_consumer


def run_job(thread_num=1, workload=[], resfile="../output.res"):
    global total_lat
    total_lat = 0
    global error_query_num
    error_query_num = 0
    workload_len = len(workload)

    global POOL
    POOL = PooledDB(
        creator=pymysql,  # 使用链接数据库的模块
        maxconnections=thread_num,  # 连接池允许的最大连接数，0和None表示不限制连接数
        mincached=0,  # 初始化时，链接池中至少创建的空闲的链接，0表示不创建
        maxcached=0,  # 链接池中最多闲置的链接，0和None不限制
        maxshared=0,
        blocking=True,  # 连接池中如果没有可用连接后，是否阻塞等待。True，等待；False，不等待然后报错
        maxusage=None,  # 一个链接最多被重复使用的次数，None表示无限制
        setsession=[],  # 开始会话前执行的命令列表。
        ping=0,
        # ping MySQL服务端，检查是否服务可用。
        host=args["host"],
        port=int(args["port"]),
        user=args["user"],
        password=args["password"],
        database=args["database"],
        charset='utf8'
    )

    main_queue = queue.Queue(maxsize=0)
    p = Producer("Producer Query", main_queue, workload)
    p.setDaemon(True)
    p.start()
    startConsumer(thread_num, main_queue)
    # 确保所有的任务都生成
    p.join()
    start = time.time()
    print("run_job开始运行,线程数：", thread_num)
    # 等待处理完所有任务
    main_queue.join()
    POOL.close()
    run_time = round(time.time() - start, 1)
    run_query_num = workload_len - error_query_num
    if run_query_num == 0:
        avg_lat = 0
        avg_qps = 0
    else:
        avg_lat = total_lat / run_query_num
        avg_qps = run_query_num / run_time
    text = "\navg_qps(queries/s): \t{}\navg_lat(s): \t{}\n".format(round(avg_qps, 4), round(avg_lat, 4))
    with open(resfile, "w+") as f:
        f.write(text)
        f.close()
    print("run_job运行结束\n脚本总耗时:{}秒,sql执行总耗时:{}秒\n共有{}条数据，执行成功{}条\n{}".format(str(run_time), str(total_lat),
                                                                            str(workload_len),
                                                                            str(run_query_num),
                                                                            text))

    return round(avg_qps, 4), round(avg_lat, 4)


def mysql_query(sql: str) -> bool:
    try:
        global POOL
        conn = POOL.connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        cursor.close()
        conn.commit()
        return True
    except Exception as error:
        print("mysql execute: " + str(error))
        return False

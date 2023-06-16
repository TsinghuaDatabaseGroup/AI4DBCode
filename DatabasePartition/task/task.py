from datetime import datetime


def my_job():
    print('定时任务my_job执行：' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


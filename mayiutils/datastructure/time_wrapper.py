#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: time_wrapper.py
@time: 2019/3/17 10:09

时间、日期操作
"""
import time
from datetime import datetime, timedelta


if __name__ == '__main__':
    mode = 4
    t = datetime.strptime('20100101', '%Y%m%d')
    t2 = datetime.strptime('20100105', '%Y%m%d')
    print(t2-t)
    print((t2-t).days)
    if mode == 3:
        # 获取当前时间
        print(datetime.now())#2019-03-17 10:25:09.004459
        # 日期加减 5天前
        print(datetime.now() - timedelta(days=5))#2019-03-12 10:27:24.957583
        print(datetime.now() - timedelta(weeks=4))#4周前 2019-02-17 10:28:45.238688
        print(datetime.now() + timedelta(weeks=4))#4周后 2019-04-14 10:28:45.238688
        now = datetime.now()
        print(now.year, now.month, now.day, now.hour, now.minute, now.second)#2019 3 17 10 33 26
        print(now.weekday())#6 代表星期日。。。
        tomorrow = now.replace(day=18)
        print(tomorrow)#2019-03-18 10:36:05.244992
        print(tomorrow.weekday())#0 代表星期一
    if mode == 2:
        """
        字符串转日期 strptime
        日期格式转字符串 strftime
        """
        t = time.strptime('20100101', '%Y%m%d')
        print(type(t))#<class 'time.struct_time'>
        print(t)# time.struct_time(tm_year=2010, tm_mon=1, tm_mday=1, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=4, tm_yday=1, tm_isdst=-1)
        print(time.strftime('%Y-%m-%d', t))#2010-01-01
        print(time.strftime('%Y-%m-%d', (2019, 3, 17, 0, 0, 0, 0, 0, 0)))#2019-03-17
        t = datetime.strptime('20100101', '%Y%m%d')
        print(type(t))#<class 'datetime.datetime'>
        print(t)#2010-01-01 00:00:00
        print(t.strftime('%Y-%m-%d'))#2010-01-01
    if mode == 1:
        """
        测算程序运行时间
        """
        start = time.clock()
        """
        Return the CPU time or real time since the start of the process or since the first call to clock(). 
        This has as much precision as the system records.
        """
        print(start)
        time.sleep(1)# Delay execution for a given number of seconds
        end = time.clock()
        print(end)
        print(end - start)#1.000145640846391
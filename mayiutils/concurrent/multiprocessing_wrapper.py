#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: multiprocessing_wrapper.py
@time: 2019/3/14 21:34

https://www.cnblogs.com/webber1992/p/6217327.html
https://www.cnblogs.com/tkqasn/p/5701230.html
"""
import multiprocessing
from multiprocessing import Process
import os
from multiprocessing import Pool,Queue
import time
from datetime import datetime
import pandas as pd

class test:
    """
    https://kexue.fm/archives/4231
    """
    def __init__(self):
        self.a = range(10)
        self.b = []

    def run(self):
        in_queue, out_queue = Queue(), Queue()
        for i in self.a:
            in_queue.put(i)

        def f(in_queue, out_queue):
            while not in_queue.empty():
                time.sleep(1)
                out_queue.put(in_queue.get()+1)

        pool = Pool(4, f, (in_queue, out_queue))

        while len(self.b) < len(self.a):
            if not out_queue.empty():
                t = out_queue.get()
                print(t)
                self.b.append(t)
        pool.terminate()


if __name__ == '__main__':
    #获取CPU核数
    #print(multiprocessing.cpu_count())#4
    mode = 1
    t = test()
    t.run()
    print(t.b)
    if mode == 0:
        def foo(i):

            print('say hi', i, os.getpid())


        for i in range(10):
            p = Process(target=foo, args=(i,))
            p.start()

        pool = multiprocessing.Pool(processes=4)
        print(datetime.now())  # 2019-05-31 10:51:58.672347
        rlist = []
        r = pd.Series()
        for _, v in r.items():
            rlist.append(pool.apply_async(foo, (v,)).get())
        pool.close()
        pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
        df = pd.DataFrame(rlist, columns=['匹配状态', '原代码', '原名称', '标准代码', '标准名称', '相似度'])
        print(df.shape)
        # print(df)
        df.to_pickle('temp/df.pkl')
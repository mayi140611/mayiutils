#!/usr/bin/python
# encoding: utf-8

import random
from matplotlib.cbook import flatten


class ListWrapper:
    """
    #主要是对python中的list的相关操作的封装
    """
    def __init__(self):
        pass

    @classmethod
    def extend(cls, list1, list2):
        """
        把list2中的元素加入到list1中
        :param list1:
        :param list2:
        :return:
        """
        list1.extend(list2)
        return list1

    @classmethod
    def sorted(cls, iterable, key=None, reverse=False):
        '''
        Return a new list containing all items from the iterable in ascending order.
        @key: 
        A custom key function can be supplied to customize the sort order
            sorted(list1,key=abs)#[0, 1, -1, 2, 3, -3, 4, -4, -5]
        @reverse: 
        the reverse flag can be set to request the result in descending order.       
        '''
        return sorted(iterable, key, reverse)
    
    @classmethod
    def shuffle(self, x, random=None):
        '''
        Shuffle list x in place, and return None.

        Optional argument random is a 0-argument function returning a
        random float in [0.0, 1.0); if it is the default None, the
        standard random.random will be used. 
        '''
        return random.shuffle(x, random)


if __name__ == '__main__':
    mode = 8
    l1 = []
    l1.append('a')
    l1[0] = -l1[0]
    print(l1)
    if mode == 7:
        #注意remove只删除第一个匹配的元素
        l1 = ['皮肤瘙痒', '瘙痒', '头痛', '瘙痒']
        l1.remove('瘙痒')
        print(l1)#['皮肤瘙痒', '头痛', '瘙痒']
    if mode == 1:
        # 生成有多个重复值的list
        l1 = ['我头疼'] * 5
        print(l1)#['我头疼', '我头疼', '我头疼', '我头疼', '我头疼']
        print(['我','头疼']*5)#['我', '头疼', '我', '头疼', '我', '头疼', '我', '头疼', '我', '头疼']
    if mode == 2:
        # 得到最大值及其所在的索引
        le = list(range(50))
        random.shuffle(le)
        print(le)
        print(max(le))
        print(le.index(max(le)))#最大值所在的索引位置
        print(le[le.index(max(le))])
    if mode == 3:
        """
        filter和map操作
        filter(func,list)
        对于list中的每个元素，如果func的返回值为True，则保留，否则去掉，形成一个新的list
        注意：filter时只是返回一个filter对象，只是一个生成器，需要list才能得到实际列表
        map(func,list)
        对于list中的每个元素，作为func的输入参数，返回值形成一个新的list
        注意：map时只是返回一个map对象，只是一个生成器，需要list才能得到实际列表
        """
        list1 = [0, 1, 2, 3, 4, -1, -5, -3, -4]
        def test1(x):
            if x > 1:
                return True
            return False
        print(filter(test1, list1))#<filter object at 0x000001AF0F44DB00>
        print(list(filter(test1, list1)))#[2, 3, 4]
        print(map(str, list1))#<map object at 0x000001C08985DB00>
        print(list(map(str, list1)))#['0', '1', '2', '3', '4', '-1', '-5', '-3', '-4']

        # map接收多个参数
        def test2(x1, x2):
            return x1+x2
        print(list(map(test2, list1, list1)))
    if mode == 4:
        """
        列表排序
        """
        list1 = [0, 1, 2, 3, 4, -1, -5, -3, -4]
        print(sorted(list1))
        # 按照绝对值从小到大排列
        print(sorted(list1, key=abs))
        # 逆序列表中的元素
        print(reversed(list1))#<list_reverseiterator object at 0x000002379E493438>
        print(list(reversed(list1)))#[-4, -3, -5, -1, 4, 3, 2, 1, 0]
        print(list(reversed('我是中国人')))#['人', '国', '中', '是', '我']
        print(list1[::-1])#[-4, -3, -5, -1, 4, 3, 2, 1, 0]

        """
        以tuple作为list的元素
        在默认情况下sort和sorted函数接收的参数是元组时，它将会先按元组的第一个元素进行排序再按第二个元素进行排序，
        再按第三个、第四个…依次排序。 我们通过一个简单的例子来了解它，以下面这个list为例：
        """
        data = [(1, 'B'), (1, 'A'), (2, 'A'), (0, 'B'), (0, 'a')]
        print(sorted(data))#[(0, 'B'), (0, 'a'), (1, 'A'), (1, 'B'), (2, 'A')]
        """
        那如何想要让它排序时不分大小写呢？
        这就要用到sort方法和sorted方法里的key参数了。 我们来看一下具体的实现：
        """
        print(sorted(data,key=lambda x:(x[0],x[1].lower())))#[(0, 'a'), (0, 'B'), (1, 'A'), (1, 'B'), (2, 'A')]
        #4 [扩展] 以dict作为list的元素
        data = [{'name': '张三', 'height': 175}, {'name': '李四', 'height': 165}, {'name': '王五', 'height': 185}]
        # 将x['height']最为返回tuple的第个一元素
        # 安装身高和姓名进行排序
        print(sorted(data, key=lambda x: (x['height'], x['name'])))#[{'name': '李四', 'height': 165}, {'name': '张三', 'height': 175}, {'name': '王五', 'height': 185}]
    if mode == 5:
        list1 = [0, 1, 2, 3, 4, -1, -5, -3, -4]
        print(list1.extend([6, 7, 8]))#None
        print(list1)#[0, 1, 2, 3, 4, -1, -5, -3, -4, 6, 7, 8]
        # 求和
        print(sum(list1))  # 18
        list2 = [6, 7, 8]
        #删除list1中同时在list2中存在的数据
        print([i for i in list1 if i not in list2])
    if mode == 6:
        """
        Python如何优雅的交错合并两个列表
        优雅的做数据处理，scipy系列库还是需要的。
        有现成matplotlib中的flatten函数可以用。
        """
        a = [1, 2, 3]
        b = [4, 5, 6]
        print(list(flatten(zip(a, b))))#[1, 4, 2, 5, 3, 6]
        a = [1, 2, 3, 7, 8]
        b = [4, 5, 6]
        print(list(flatten(zip(a, b))))##[1, 4, 2, 5, 3, 6]



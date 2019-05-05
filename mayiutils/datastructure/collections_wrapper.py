#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: collections_wrapper.py
@time: 2019/3/8 20:47
"""
from collections import namedtuple
from collections import deque
from collections import OrderedDict
from collections import Counter
from collections import defaultdict


class CollectionsWrapper:
    """
    collections是Python内建的一个集合模块，提供了许多有用的集合类。
    参考：
    https://www.liaoxuefeng.com/wiki/001374738125095c955c1e6d8bb493182103fac9270762a000/001411031239400f7181f65f33a4623bc42276a605debf6000
    """
    def test(self):
        pass


if __name__ == '__main__':
    mode = 3
    if mode == 4:
        """
        Counter是一个简单的计数器，例如，统计字符出现的个数：
        Counter实际上也是dict的一个子类
        Counter 对象有一个叫做 elements() 的方法，其返回的序列中，依照计数重复元素相同次数，元素顺序是无序的
        """
        c = Counter()
        for ch in 'programming':
            c[ch] = c[ch] + 1
        print(c)#Counter({'r': 2, 'g': 2, 'm': 2, 'p': 1, 'o': 1, 'a': 1, 'i': 1, 'n': 1})
        print(Counter('programming'))#Counter({'r': 2, 'g': 2, 'm': 2, 'p': 1, 'o': 1, 'a': 1, 'i': 1, 'n': 1})
        print(dict(Counter('programming')))#{'p': 1, 'r': 2, 'o': 1, 'g': 2, 'a': 1, 'm': 2, 'i': 1, 'n': 1}
        print(Counter('programming').most_common(3))#[('r', 2), ('g', 2), ('m', 2)]
        print(Counter('programming').elements())#<itertools.chain object at 0x00000283A45B5780>
        print(list(Counter('programming').elements()))#['p', 'r', 'r', 'o', 'g', 'g', 'a', 'm', 'm', 'i', 'n']
    if mode == 3:
        """
        OrderedDict的Key会按照插入的顺序排列，不是Key本身排序
        使用dict时，如果引用的Key不存在，就会抛出KeyError。如果希望key不存在时，返回一个默认值，就可以用defaultdict
        """
        od = OrderedDict([('a', 1), ('b', 2), ('c', 3)])
        print(od)#OrderedDict([('a', 1), ('b', 2), ('c', 3)])

        dd = defaultdict(lambda: 'N/A')
        dd['key1'] = 'abc'
        print(dd['key1'])#abc
        print(dd['key2'])#N/A

        dd = defaultdict(int)
        print(dd['a'])#0
    if mode == 1:
        """
        namedtuple是一个函数，它用来创建一个自定义的tuple对象，并且规定了tuple元素的个数，并可以用属性而不是索引来引用tuple的某个元素。

        这样一来，我们用namedtuple可以很方便地定义一种数据类型，它具备tuple的不变性，又可以根据属性来引用，使用十分方便。
        
        可以验证创建的Point对象是tuple的一种子类：
        """
        Point = namedtuple('Point', ['x', 'y'])
        p = Point(1, 2)
        print(p.x, p.y)
        print(isinstance(p, Point))
        print(isinstance(p, tuple))
    if mode == 2:
        """
        使用list存储数据时，按索引访问元素很快，但是插入和删除元素就很慢了，因为list是线性存储，数据量大的时候，插入和删除效率很低。
        deque是为了高效实现插入和删除操作的双向列表，适合用于队列和栈：
        """
        q = deque(['a', 'b', 'c'])
        q.append('x')
        q.appendleft('y')
        print(q)









#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: Python面向对象example.py
@time: 2019-06-05 13:46

 Python 面向对象编程的最佳实践
https://mp.weixin.qq.com/s/oHK-Y4lOeaQCFtDWgqXxFA

pip install attrs cattrs

有了 attrs 库，我们就可以非常方便地定义各个对象了，另外对于 JSON 的转化，可以进一步借助 cattrs 这个库，非常有帮助。

attrs能将你从繁综复杂的实现上解脱出来，享受编写 Python 类的快乐。它的目标就是在不减慢你编程速度的前提下，帮助你来编写简洁而又正确的代码。

在 attr 这个库里面有两个比较常用的组件叫做 attrs 和 attr，前者是主要用来修饰一个自定义类的，后者是定义类里面的一个字段的。

attrs 里面修饰了 Color 这个自定义类，然后用 attrib 来定义一个个属性，同时可以指定属性的类型和默认值。

实际上，主要是 attrs 这个修饰符起了作用，然后根据定义的 attrib 属性自动帮我们实现了
 __init__、__repr__、__eq__、__ne__、__lt__、__le__、__gt__、__ge__、__hash__ 这几个方法。

 别名使用
 s = attributes = attrs
ib = attr = attrib

验证器
attrs 库里面还给我们内置了好多 Validator，比如判断类型，这里我们再增加一个属性 age，必须为 int 类型：

age = attrib(validator=validators.instance_of(int))
"""
from attr import attrs, attrib, fields
from cattr import unstructure, structure


class Color(object):
    """
    Color Object of RGB
    """
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    def __repr__(self):
        """
        在 Python 里面想要定义某个对象本身的打印输出结果的时候，需要实现它的 __repr__ 方法
        :return:
        """
        return f'{self.__class__.__name__}(r={self.r}, g={self.g}, b={self.b})'

    def __lt__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (self.r, self.g, self.b) < (other.r, other.g, other.b)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (self.r, self.g, self.b) == (other.r, other.g, other.b)


@attrs
class Color1(object):
    r = attrib(type=int, default=0)
    g = attrib(type=int, default=0)
    b = attrib(type=int, default=0)


@attrs
class Point(object):
    x = attrib()
    y = attrib()


def is_valid_gender(instance, attribute, value):
    if value not in ['male', 'female']:
        raise ValueError(f'gender {value} is not valid')


@attrs
class Person(object):
    name = attrib()
    gender = attrib(validator=is_valid_gender)


if __name__ == '__main__':
    mode = 5
    if mode == 5:
        point = Point(x=1, y=2)
        json = unstructure(point)
        print('json:', json)
        obj = structure(json, Point)
        print('obj:', obj)
    if mode == 4:
        print(Person(name='Mike', gender='male'))
        print(Person(name='Mike', gender='mlae'))
    if mode == 3:
        print(fields(Point))
        """
        (Attribute(name='x', default=NOTHING, validator=None, repr=True, cmp=True, hash=None, init=True, metadata=mappingproxy({}), type=None, converter=None, kw_only=False), 
        Attribute(name='y', default=NOTHING, validator=None, repr=True, cmp=True, hash=None, init=True, metadata=mappingproxy({}), type=None, converter=None, kw_only=False))
name：属性的名字，是一个字符串类型。
default：属性的默认值，如果没有传入初始化数据，那么就会使用默认值。如果没有默认值定义，那么就是 NOTHING，即没有默认值。
validator：验证器，检查传入的参数是否合法。
init：是否参与初始化，如果为 False，那么这个参数不能当做类的初始化参数，默认是 True。
metadata：元数据，只读性的附加数据。
type：类型，比如 int、str 等各种类型，默认为 None。
converter：转换器，进行一些值的处理和转换器，增加容错性。
kw_only：是否为强制关键字参数，默认为 False。
        """

    if mode == 2:
        c1 = Color1(254, 254, 255)
        c2 = Color1(254, 255, 255)
        c3 = Color1(254, 255, 255)
        print(c1)  # Color(r=255, g=255, b=255)

        print(c1 < c2)  # True
        print(c3 == c2)  # True
    if mode == 1:
        """
        """
        c1 = Color(254, 254, 255)
        c2 = Color(254, 255, 255)
        c3 = Color(254, 255, 255)
        print(c1)  # Color(r=255, g=255, b=255)

        print(c1 < c2)  # True
        print(c3 == c2)  # True


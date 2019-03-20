#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: xml_wrapper.py
@time: 2019/3/19 20:05

https://www.cnblogs.com/xiaobingqianrui/p/8405813.html
"""
import xml.dom.minidom as xmldom


if __name__ == '__main__':
    # 得到文档对象
    xmlfilepath = 'news_sohusite.xml'
    domobj = xmldom.parse(xmlfilepath)
    print("xmldom.parse:", type(domobj))

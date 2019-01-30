#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: py2neo_wrapper.py
@time: 2019/1/27 22:13
py2neo==3.1.2
neo4j-driver==1.6.0
"""
from py2neo import Graph


class Py2NeoWrapper:
    @classmethod
    def grapth(cls):
        return Graph("http://localhost:7474",user='neo4j',password='123')

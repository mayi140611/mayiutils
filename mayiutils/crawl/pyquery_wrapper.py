#!/usr/bin/python
# encoding: utf-8
from pyquery import PyQuery as pq


class PyQueryWrapper(object):
    def __init__(self, text):
        self._root = pq(text)

    @property
    def root(self):
        '''
        返回文档的根
        '''
        return self._root
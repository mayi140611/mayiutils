#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: es_wrapper.py
@time: 2019/1/21 18:22
"""
#pip install elasticsearch
from elasticsearch import Elasticsearch


class EsWrapper:
    def __init__(self, hosts):
        self._es = Elasticsearch(hosts)

    def createIndex(self, index):
        '''
        创建索引库
        :param name:
        :return:
        '''
        #注意这里我们的代码里面使用了 ignore 参数为 400，
        # 这说明如果返回结果是 400 的话，就忽略这个错误不会报错，程序不会执行抛出异常。
        return self._es.indices.create(index=index, ignore=400)
    def deleteIndex(self, index):
        '''
        删除索引库
        :param name:
        :return:
        '''
        return self._es.indices.delete(index=index, ignore=[400, 404])
    def insertDoc(self, index, doc_type, body):
        '''
        插入一条记录
        可以使用 index() 方法来插入数据，但与 create() 不同的是，create() 方法需要我们指定 id 字段来唯一标识该条数据，
        而 index() 方法则不需要，如果不指定 id，会自动生成一个 id
        :param index:
        :param doc_type:
        :return:
        '''
        return self._es.index(index=index, doc_type=doc_type, body=body)

    def insertDocById(self, index, doc_type, id, body):
        '''
        插入一条记录
        :param index:
        :param doc_type:
        :return:
        '''
        return self._es.create(index=index, doc_type=doc_type, id=id, body=body)

    def updateDocById(self, index, doc_type, id, body):
        '''

        :param index:
        :param doc_type:
        :param id:
        :param body:
        :return:
        '''
        return self._es.update(index=index, doc_type=doc_type, id=id, body=body)

    def deleteDocById(self, index, doc_type, id):
        '''

        :param index:
        :param doc_type:
        :param id:
        :param body:
        :return:
        '''
        return self._es.delete(index=index, doc_type=doc_type, id=id)

    def putMapping(self, index, doc_type, body):
        '''

        :param index:
        :param doc_type:
        :param body:
        :return:
        '''
        return self._es.indices.put_mapping(doc_type, body, index )
    def search(self, index, doc_type, body=None):
        return self._es.search(index, doc_type, body)
#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: es_wrapper.py
@time: 2019/1/21 18:22
#pip install elasticsearch==6.3.1
"""
from elasticsearch import Elasticsearch


class EsWrapper:
    """
    es中的DSL（领域查询语言），示例：
    {
        'query': {'match_all': {}},
        'sort': [{'date': 'desc'}],
        'from': 1,
        'size': 2,
        '_source': ['title', 'date']
    }
    查询语法解析， query表示查询的定义， match_all部分表示查询类型。
    如果size没有指定， 默认是10.

    查询类型有：
    match_all
    match：match查询的时候,elasticsearch会根据你给定的字段提供合适的分析器
        如："match": { "address": "mill lane" }
            返回地址中包含term为“mill”或“lane”的所有帐户
    match_phrase： 是match（match_phrase）的变体，
        "match_phrase": { "address": "mill lane" }
            它返回地址中包含短语“mill lane”的所有帐户
    bool查询：
         must子句指定所有必须为true的查询才能将文档视为匹配项
         should子句指定了一个查询列表，其中任何一个查询必须为true才能将文档视为匹配项
         must_not子句指定了一个查询列表，对于文档而言，这些查询都不能为true
        {
          "query": {
            "bool": {
              "must": [
                { "match": { "address": "mill" } },
                { "match": { "address": "lane" } }
              ]
            }
          }
        }
    filter过滤：过滤文档时， 查询不需要产生分数。 当使用filter时， Elasticsearch不计算分数。
        filter过滤子句允许使用查询来限制与其他子句匹配的文档， 而不会更改计算得分的方式。
        {
          "query": {
            "bool": {
              "must": { "match_all": {} },
              "filter": {
                "range": {
                  "balance": {
                    "gte": 20000,
                    "lte": 30000
                  }
                }
              }
            }
          }
        }
    Aggregations 聚合提供了从数据中分组和提取统计信息的功能
        按state对所有的账户进行分组， 然后按计数降序（默认）排序的前10个state
        GET /bank/_search
        {
          "size": 0,
          "aggs": {
            "group_by_state": {
              "terms": {
                "field": "state.keyword"
              }
            }
          }
        }
        其中"size": 0表示不显示搜索结果， 只显示聚合结果。
        下面的示例按州计算平均账户余额
        {
          "size": 0,
          "aggs": {
            "group_by_state": {
              "terms": {
                "field": "state.keyword"
              },
              "aggs": {
                "average_balance": {
                  "avg": {
                    "field": "balance"
                  }
                }
              }
            }
          }
        }
    """
    def __init__(self, hosts):
        self._es = Elasticsearch(hosts)

    def createIndex(self, indexname, body=None, params=None):
        """
        创建索引库
        PUT /customer?pretty
        :param indexname: 索引名
            索引名规范：仅限小写；不能以-、_、+开头；不能超过255字节
        :param body:
        设置分片数和副本数（默认是5个分片，1个副本
            {
              "settings" : {
                "index" : {
                  "number_of_shards" : 3,
                  "number_of_replicas" : 2
                }
              }
            }
        创建索引时还可以提供一个type的mapping
            {
              "settings" : {
                "number_of_shards" : 1
              },
              "mappings" : {
                "_doc" : {
                  "properties" : {
                  "field1" : { "type" : "text" }
                  }
                }
              }
            }
        :param params:
        :return:
        """
        '''
        :return:
        '''
        #注意这里我们的代码里面使用了 ignore 参数为 400，
        # 这说明如果返回结果是 400 的话，就忽略这个错误不会报错，程序不会执行抛出异常。
        return self._es.indices.create(index=indexname, body=body, ignore=400)

    def deleteIndex(self, index):
        '''
        删除索引库
        DELETE /customer?pretty
        :param name:
        :return:
        '''
        return self._es.indices.delete(index=index, ignore=[400, 404])

    def insertDoc(self, index, doc_type, body, id=None, params=None):
        '''
        插入一条记录
        Adds or updates a typed JSON document in a specific index, making it searchable.
        可以使用 index() 方法来插入数据，但与 create() 不同的是，create() 方法需要我们指定 id 字段来唯一标识该条数据，
        而 index() 方法则不需要，如果不指定 id，会自动生成一个 id
        :param index:
        :param doc_type:
        :return:
        '''
        return self._es.index(index, doc_type, body, id, params)

    def insertDocById(self, index, doc_type, id, body):
        '''
        插入一条记录
        Adds a typed JSON document in a specific index, making it searchable.
        Behind the scenes this method calls index(..., op_type='create')
        :param index:
        :param doc_type:
        :return:
        '''
        return self._es.create(index=index, doc_type=doc_type, id=id, body=body)

    def get(self, index, doc_type, id, params=None):
        """
        Get a typed JSON document from the index based on its id.

        GET /customer/_doc/1?pretty
        :return:
        """
        return self._es.get(index, doc_type, id, params)


    def updateDocById(self, index, doc_type, id, body):
        '''
        POST /customer/_doc/1/_update?pretty
        :param index:
        :param doc_type:
        :param id:
        :param body:
        :return:
        '''
        return self._es.update(index=index, doc_type=doc_type, id=id, body=body)

    def deleteDocById(self, index, doc_type, id):
        '''
        DELETE /customer/_doc/2?pretty
        :param index:
        :param doc_type:
        :param id:
        :param body:
        :return:
        '''
        return self._es.delete(index=index, doc_type=doc_type, id=id)

    def putMapping(self, index, doc_type, body):
        '''
        Register specific mapping definition for a specific type.
        :param index:
        :param doc_type:
        :param body:
        :return:
        '''
        return self._es.indices.put_mapping(doc_type, body, index )

    def search(self, index, doc_type=None, body=None):
        """
        Execute a search query and get back search hits that match the query.
        es.search(index='news', doc_type='politics')
        #查询出所有条目
        GET /news/politics/_search  等价于
        query = {
            'query': {'match_all': {}}
        }

        :param index:
        :param doc_type:
        :param body: request body
            执行搜索有两种方法： request URI、request body
            es中有多种查询类型：
                match_all：查询所有数据


        :return:
            took – Elasitcsearch执行搜索的时间
            timed_out – 搜索是否超时
            _shards – 搜索了多少分片，以及搜索成功/失败分片的计数
            hits – 搜索结果
            hits.total – 符合搜索条件的文档总数
            hits.hits – 搜索结果（默认为前10个文档）
            hits.sort 结果的排序键（如果按分数排则没有此项）
            {
              "took": 63,
              "timed_out": false,
              "_shards": {
                "total": 5,
                "successful": 5,
                "skipped": 0,
                "failed": 0
              },
              "hits": {
                "total": 4,
                "max_score": 1,
                "hits": [
                  {
                    "_index": "news",
                    "_type": "politics",
                    "_id": "ATIicGgBBbbIOC8E9Gke",
                    "_score": 1,
                    "_source": {
                      "title": "公安部：各地校车将享最高路权",
                      "url": "http://www.chinanews.com/gn/2011/12-16/3536077.shtml",
                      "date": "2011-12-16"
                    }
                  },
        """
        return self._es.search(index, doc_type, body)
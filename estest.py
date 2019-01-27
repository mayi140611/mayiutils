#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: estest.py
@time: 2019/1/21 18:26
"""
from mayiutils.es_wrapper import EsWrapper
import json

if __name__ == '__main__':
    es = EsWrapper(['h1', 'h2'])
    # result = es.createIndex('news')
    # result = es.deleteIndex('news')
    # data = {'title': '美国留给伊拉克的是个烂摊子吗1', 'url': 'http://view.news.qq.com/zt2011/usa_iraq/index.htm'}
    # result = es.insertDocById('news','politics',id=1, body=data)
    # result = es.insertDoc('news','politics', body=data)
    # data = {
    #     'title': '美国留给伊拉克的是个烂摊子吗',
    #     'url': 'http://view.news.qq.com/zt2011/usa_iraq/index.htm',
    #     'date': '2011-12-16'
    # }
    # result = es.updateDocById('news', 'politics', id=2, body=data)
    # result = es.deleteDocById('news', 'politics', id=1)
    #
    datas = [
        {
            'title': '美国留给伊拉克的是个烂摊子吗',
            'url': 'http://view.news.qq.com/zt2011/usa_iraq/index.htm',
            'date': '2011-12-16'
        },
        {
            'title': '公安部：各地校车将享最高路权',
            'url': 'http://www.chinanews.com/gn/2011/12-16/3536077.shtml',
            'date': '2011-12-16'
        },
        {
            'title': '中韩渔警冲突调查：韩警平均每天扣1艘中国渔船',
            'url': 'https://news.qq.com/a/20111216/001044.htm',
            'date': '2011-12-17'
        },
        {
            'title': '中国驻洛杉矶领事馆遭亚裔男子枪击 嫌犯已自首',
            'url': 'http://news.ifeng.com/world/detail_2011_12/16/11372558_0.shtml',
            'date': '2011-12-18'
        }
    ]
    # for data in datas:
    #     es.insertDoc('news','politics',data)
    # result = es.search(index='news', doc_type='politics')
    # print(result)
    dsl = {
        'query': {
            'match': {
                'title': '中国 领事馆'
            }
        },
        "highlight": {
            "pre_tags": ["<red>"],
            "post_tags": ["</red>"],
            "fields": {
                "title": {}
            }
        }
    }

    result = es.search(index='news', doc_type='politics', body=dsl)
    print(json.dumps(result, indent=2, ensure_ascii=False))





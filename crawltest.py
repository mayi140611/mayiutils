#!/usr/bin/python
# encoding: utf-8

from mayiutils.pickle_wrapper import pickle_wrapper
from mayiutils.crawl import requests_wrapper as rw
from mayiutils.crawl import pyquery_wrapper as pw
from mayiutils.db.pymongo_wrapper import PyMongoWrapper

def parseList(url, selectors, saveTo='mongodb', dbname=None, tablename=None, filepath=None,
              nextTag='下一页', pretag=''):
    '''
    爬取页面的列表[title:url,...]
    :param url:  带爬取页面的url
    :param selectors:  页面中要爬取元素的选择器
    :param saveTo:  爬取的列表的保存位置
        可以是mongodb，可以是pkl
    :return:
    '''
    r1 = rw.RequestsWrapper(url)
    html = r1.getText()
    d = pw.PyQueryWrapper(html).root
    #用一个字典保存一条爬取的结果
    if saveTo == 'mongodb':
        pmw = PyMongoWrapper('h1')
        table = pmw.getCollection(dbname, tablename)
        pmw.setUniqueIndex(dbname, tablename, 'title')
        for i in range(len(d(selectors))):
            try:
                d1 = dict()
                d1['title'] = d(selectors).eq(i).text().strip()
                d1['url'] = d(selectors).eq(i).attr.href
                print(d1)
                table.insert_one(d1)
            except Exception as e:
                print(e, d1)
        #获取“下一页”所在的链接地址
        nexturl = d('.bigPage > :contains({})'.format(nextTag)).eq(0).attr.href
        print(nexturl)
        if nexturl:
            nexturl = '{}{}'.format(pretag, d('.bigPage > :contains({})'.format(nextTag)).eq(0).attr.href)
            parseList(nexturl, selectors, saveTo, dbname, tablename, filepath,
                  nextTag, pretag)


def parseDetail(url, selectors):
    """
    爬取页面详情
    :param url: 待爬取页面的url
    :param selectors: dict。爬取元素的选择器
    :return: 解析好的字典
    """
    r1 = rw.RequestsWrapper(url)
    html = r1.getText()
    d = pw.PyQueryWrapper(html).root
    d1 = dict()
    d1['url'] = url
    try:
        for k, v in selectors.items():
            if k.endswith("List"):#如果以List结果，说明要爬取的是一个list
                if k == "mainblockList":#爬取主模块内容
                    for i in range(len(d(v[0]))):
                        tag = d(v[0])(v[1]).eq(i).text().strip()
                        content = d(v[0])(v[2]).eq(i).text().strip()
                        d1[tag] = content
                else:
                    l1 = list()
                    for i in range(len(d(v))):
                        l1.append(d(v).eq(i).text().strip())
                    d1[k] = l1
            else:
                d1[k] = d(v).eq(0).text().strip()

    except Exception as e:
        print(e, k, d1)
    return d1

if __name__ == '__main__':
    pkw = pickle_wrapper()
    # pkw.dump2file(list('123'), 'abc.pkl')
    # parseList('https://www.baikemy.com/disease/list/0/0?pageIndex=1&pageCount=82',
    #               'li.ccjb_jbli_li > a:nth-child(1)',
    #               dbname='mybk',
    #               tablename='jbproduct',
    #               pretag='https://www.baikemy.com'
    #               )

    pmw = PyMongoWrapper('h1')
    # table = pmw.getCollection('mybk', 'jbproduct')
    table1 = pmw.getCollection('mybk', 'jbproductdetail')
    # urls = list(pmw.findAll(table,fieldlist=['url']))

    # pkw.dump2file(urls, 'urls.pkl')
    urls = pkw.loadfromfile('urls.pkl')
    count = 8842
    for url in urls[8842:]:
        print('{}--{}'.format(url,count))
        url1 = url['url'][(url['url'].rfind('/')+1) :]
        # print(url, url1)
        # break
        r = parseDetail('https://www.baikemy.com/disease/detail/{}/1'.format(url1),
                        {"title":'.jb-head-left-text',
                         "expertList":'.jbct-ctzz-center > ul:nth-child(1) > li',
                         "mainblockList":['div.lemma-main-content',
                                          'div.headline-1 > span.headline-content',
                                           'div.para']
                             }
                          )
        if '病因' in r or '临床表现' in r:
            r['isdisease']='true'
        else:
            r['isdisease']='false'
            print(r)
        table1.insert_one(r)
        count += 1






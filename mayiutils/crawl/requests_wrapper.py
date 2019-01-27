#!/usr/bin/python
# encoding: utf-8
import requests


class RequestsWrapper(object):

    def __init__(self, url):
        self._url = url
        self._headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.79 Safari/537.36 Maxthon/5.2.1.6000'
        }
    def getResponse(self, encoding='utf8', data=None, allow_redirects=False, proxies=None):
        r = requests.get(self._url,headers=self._headers,allow_redirects=allow_redirects,proxies=proxies)
        r.encoding=encoding
        return r
    
    def getText(self, encoding='utf8', data=None):
        return self.getResponse(encoding, data).text
    
#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: flask_wrapper.py
@time: 2019/3/8 18:20
"""
from flask import Flask
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'hello123'


if __name__ == '__main__':
    app.run()
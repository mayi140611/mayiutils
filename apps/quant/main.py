#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: main.py
@time: 2019-06-19 15:42
"""
if __name__ == '__main__':
    while True:
        print('please select: 1) data explore; 2) train; 3) test')
        a = input()
        if a == 'q':
            print(f'bye! {a}')
            break
        print(f'hi {a}')
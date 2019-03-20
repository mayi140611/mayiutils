#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: tqdm_wrapper.py
@time: 2019/3/20 9:36

Iterable to decorate with a progressbar. Leave blank to manually manage the updates.
"""
import tqdm


if __name__ == '__main__':
    for a in tqdm(rlist):
        pass
    #100%|██████████| 413178/413178 [01:47<00:00, 3847.60it/s]
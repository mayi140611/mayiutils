#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: rs.py
@time: 2019/3/16 18:32

采用无监督的方式做， 通过估算正态分布
"""
from surprise import NormalPredictor
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
import pandas as pd


if __name__ == '__main__':
    traindf = pd.read_csv('D:/Desktop/DF/portrait/train_dataset.csv')
    print(traindf.columns)
    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(422, 719))
    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(traindf[['用户编码', '当月网购类应用使用次数', '信用分']], reader)
    algo = NormalPredictor()
    # perf = cross_validate(algo, data, cv=5, measures=['RMSE', 'MAE'],  verbose=True)
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    r = []
    for line in traindf.itertuples():
        # print(line[1], line[23])
        pred = algo.predict(line[1], line[23], r_ui=4, verbose=True)
        print(type(pred.est), pred.est)
        r.append(round(pred.est))
    traindf['r'] = r
    traindf.to_csv('D:/Desktop/DF/portrait/train_dataset1.csv')


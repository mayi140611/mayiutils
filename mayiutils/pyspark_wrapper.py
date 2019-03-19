#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: pyspark_wrapper.py
@time: 2019/3/19 11:18
"""
import pyspark
from pyspark.sql import SparkSession


if __name__ == '__main__':
    #生成SparkSession实例
    spark = SparkSession.builder \
         .master("spark://h1:7077") \
         .appName("sparkmlhw02") \
         .config("spark.some.config.option", "some-value") \
         .getOrCreate()
    df1 = spark.read.csv("file:///home/ian/code/data/sparkml/doc_class.dat", sep='|', header=True)
    df1.show(5)
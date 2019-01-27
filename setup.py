#!/usr/bin/python
# encoding: utf-8


from distutils.core import setup
from setuptools import find_packages
setup(
    name="mayiutils",
    version="0.1",
    description="工具类", author="mayi140611",
    packages=find_packages(include=['mayiutils', 'mayiutils.*'])
)

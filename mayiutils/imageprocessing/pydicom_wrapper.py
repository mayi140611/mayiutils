#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: pydicom_wrapper.py
@time: 2019/2/22 13:05
pydicom-1.2.2
"""
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc


class PydicomWrapper:
    """
    Pydicom是一个处理DICOM文件的纯Python软件包。
    """
    @classmethod
    def readFile(cls, filepath):
        """
        Read and parse a DICOM dataset stored in the DICOM File Format.
        :param filepath:
        :return:
        """
        return pydicom.read_file(filepath)


if __name__ == '__main__':
    in_path = 'test.dcm'
    out_path = 'output.jpg'
    ds = pydicom.read_file(in_path)  # 读取.dcm文件
    img = ds.pixel_array  # 提取图像信息
    print(type(img), img)
    print(np.max(img, axis=0))
    print(img.shape)
    plt.imshow(img)
    plt.show()
    scipy.misc.imsave(out_path, img)#转换为jpg格式存储
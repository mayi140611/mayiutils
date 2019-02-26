#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: opencv_wrapper.py
@time: 2019/2/24 11:41

opencv-python==3.4.4.19
"""
import cv2


class OpencvWrapper:
    @classmethod
    def imread(cls, filename, flags=cv2.IMREAD_COLOR):
        """
        读入一副图片
        :param filename:
        :param flags:
            cv2.IMREAD_COLOR：默认参数，读入一副彩色图片，忽略alpha通道
            cv2.IMREAD_GRAYSCALE：读入灰度图片(只有一个通道）
            cv2.IMREAD_UNCHANGED：顾名思义，读入完整图片，包括alpha通道
        :return:
        """
        return cv2.imread(filename, flags)

    @classmethod
    def cvtColor(cls, img, flags):
        """
        颜色空间转换
        #彩色图像转为灰度图像
        img2 = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        #灰度图像转为彩色图像
        img3 = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # cv2.COLOR_X2Y，其中X,Y = RGB, BGR, GRAY, HSV, YCrCb, XYZ, Lab, Luv, HLS
        :param img:
        :param flags:
        :return:
        """
        return cv2.cvtColor(img, flags)

    @classmethod
    def imshow(cls, mat, winname='winname1'):
        """
        显示图像
        :param winname:显示图像的窗口的名字
        :param mat:要显示的图像（imread读入的图像），窗口大小自动调整为图片大小
        :return:
        """
        return cv2.imshow(winname, mat)

    @classmethod
    def waitKey(cls, delay=0):
        """
        等待键盘输入，单位为毫秒，即等待指定的毫秒数看是否有键盘输入，若在等待时间内按下任意键则返回按键的ASCII码，程序继续运行。
        若没有按下任何键，超时后返回-1。参数为0表示无限等待。
        不调用waitKey的话，窗口会一闪而逝，看不到显示的图片。
        :param delay:
        :return:
        """
        return cv2.waitKey(delay)

    @classmethod
    def destroyAllWindows(cls):
        """
        销毁所有窗口
        :return:
        """
        return cv2.destroyAllWindows()

    @classmethod
    def destroyWindow(cls, winname):
        """
        销毁指定窗口
        :param winname:
        :return:
        """
        return cv2.destroyWindow(winname)

    @classmethod
    def imwrite(cls, filename, img, params=None):
        """
        保存一个图像
        :param filename:要保存的文件名
        :param img:要保存的图像
        :param params:
            针对特定的格式：对于JPEG，其表示的是图像的质量，用0 - 100的整数表示，默认95;
            对于png ,第三个参数表示的是压缩级别。默认为3.
            注意:

            cv2.IMWRITE_JPEG_QUALITY类型为 long ,必须转换成 int
            cv2.IMWRITE_PNG_COMPRESSION, 从0到9 压缩级别越高图像越小。
            cv2.imwrite('1.png',img, [int( cv2.IMWRITE_JPEG_QUALITY), 95])
            cv2.imwrite('1.png',img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        :return:
        """
        return cv2.imwrite(filename, img, params)

    @classmethod
    def flip(cls, img, flipcode=0):
        """
        翻转图像，flipcode控制翻转效果。
        :param img:
        :param flipcode:
            flipcode = 0：沿x轴翻转
            flipcode > 0：沿y轴翻转
            flipcode < 0：x,y轴同时翻转
        :return:
        """
        return cv2.flip(img, flipcode)

    @classmethod
    def resize(cls, img, shape):
        """
        缩放图片
        :param img:
        :param shape:  元组(宽度,高度)
        :return:
        """
        return cv2.resize(img, shape)


if __name__ == '__main__':
    img = OpencvWrapper.imread('output.jpg')
    print(type(img), img.shape)
    # print(img[:, :, 0])
    # imgflip = OpencvWrapper.flip(img, 0)
    # OpencvWrapper.imshow(imgflip)
    # OpencvWrapper.waitKey()

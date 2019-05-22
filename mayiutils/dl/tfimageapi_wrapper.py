#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: tfimageapi_wrapper.py
@time: 2019/3/12 13:46
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    image_raw_data = tf.gfile.FastGFile("../../imageprocessing/img1.jpg", 'rb').read()

    with tf.Session() as sess:
        # 将图像使用jpeg的格式解码从而得到图像对应的三维矩阵，
        # tf.image.decode_png函数对png格式的图像进行解码，
        # 解码后的结果是一个tensor
        # image_data = tf.image.decode_jpeg(image_raw_data)
        image_data = tf.image.decode_png(image_raw_data)
        # print(image_data.eval())
        # plt.imshow(image_data.eval())
        # plt.show()

        resized = tf.image.resize_images(image_data, [100, 100], method=3)
        # print(resized.dtype)#<dtype: 'float32'>
        # print(image_data.get_shape())#(?, ?, ?)
        # plt.imshow(resized.eval())
        # plt.show()

        # TensorFlow的函数处理图片后存储的数据是float32格式的，需要转换成uint8才能正确打印图片
        resized = np.asarray(resized.eval(), dtype='uint8')
        img_data = tf.image.convert_image_dtype(resized, dtype=tf.uint8)
        # plt.imshow(img_data.eval())
        # plt.show()

        # encoded_image = tf.image.encode_jpeg(img_data)
        # with tf.gfile.GFile("../../../tmp/re3-image.jpeg", "wb") as f:
        #     f.write(encoded_image.eval())

        transposed = tf.image.transpose_image(image_data)
        # plt.imshow(transposed.eval())
        # plt.show()

        filpped = tf.image.random_flip_up_down(image_data)
        # plt.imshow(filpped.eval())
        # plt.show()

        adjusted = tf.image.random_brightness(image_data, 0.5)
        # plt.imshow(adjusted.eval())
        # plt.show()

        adjusted_constrast = tf.image.random_contrast(image_data, 0, 20)
        # plt.imshow(adjusted_constrast.eval())
        # plt.show()

        adjusted_hue = tf.image.random_hue(image_data, 0.5)
        # plt.imshow(adjusted_hue.eval())
        # plt.show()

        adjusted_saturation = tf.image.adjust_saturation(image_data, -5)
        plt.imshow(adjusted_saturation.eval())
        plt.show()

        adjusted_whitening = tf.image.per_image_standardization(image_data)
        # 有负值，所以画图画不出来
        # plt.imshow(adjusted_whitening.eval())
        # plt.show()

        # 缩小图像，让标注框更清晰
        resized_image = tf.image.resize_images(image_data, (200, 200), method=1)
        batched = tf.expand_dims(tf.image.convert_image_dtype(resized_image, tf.float32), 0)
        boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
        result = tf.image.draw_bounding_boxes(batched, boxes)
        # resized_image_show = np.asarray(result.eval(), dtype=tf.uint8)
        result1 = tf.squeeze(result, axis=0)
        # plt.imshow(result1.eval())
        # plt.show()

        # 随机截取图像上有信息含量的部分，也可以提高模型健壮性
        # 此函数为图像生成单个随机变形的边界框。函数输出的是可用于裁剪原始图像的单个边框。
        # 返回值为3个张量：begin，size和 bboxes。前2个张量用于 tf.slice 剪裁图像。
        # 后者可以用于 tf.image.draw_bounding_boxes 函数来画出边界框。
        boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
        print(tf.shape(image_data))
        begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(tf.shape(image_data), bounding_boxes=boxes,
                                                                            min_object_covered=0.1)
        batched = tf.expand_dims(tf.image.convert_image_dtype(image_data, tf.float32), 0)
        image_with_box = tf.image.draw_bounding_boxes(batched, bbox_for_draw)
        distorted_image = tf.slice(image_data, begin, size)
        plt.imshow(distorted_image.eval())
        plt.show()

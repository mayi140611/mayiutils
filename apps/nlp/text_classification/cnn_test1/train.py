#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: train.py
@time: 2019-05-05 14:33
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import time
import datetime
from apps.nlp.text_classification.cnn_test1 import data_prepare
from tensorflow.contrib import learn
import matplotlib.pyplot as plt


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "/Users/luoyonggui/Documents/datasets/nlp/classification/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "/Users/luoyonggui/Documents/datasets/nlp/classification/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text, y = data_prepare.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    print(f'句子的最大长度是： {max_document_length}')#56
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    print(type(vocab_processor))#<class 'tensorflow.contrib.learn.python.learn.preprocessing.text.VocabularyProcessor'>
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    print(x.shape)#(10662, 56)
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))#Vocabulary Size: 18758
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))#Train/Dev split: 9596/1066
    return x_train, y_train, vocab_processor, x_dev, y_dev


def train():
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    model = keras.Sequential()
    # model.add(keras.layers.Embedding(len(vocab_processor.vocabulary_), FLAGS.embedding_dim, input_shape=(56, 1)))
    #
    # model.add(keras.layers.Conv2D(32, 3, activation='relu'))
    # model.add(keras.layers.AveragePooling2D())
    model.add(keras.layers.Embedding(len(vocab_processor.vocabulary_), FLAGS.embedding_dim, input_shape=(56, )))
    model.add(keras.layers.Conv1D(2, 3, activation='relu'))
    model.add(keras.layers.AveragePooling1D())
    model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(2, activation=tf.nn.softmax))
    model.summary()
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  # loss='binary_crossentropy',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # history = model.fit(x_train[:, :, np.newaxis],
    #                     y_train,
    #                     epochs=40,
    #                     batch_size=512,
    #                     validation_data=(x_dev[:, :, np.newaxis], y_dev),
    #                     verbose=1)
    history = model.fit(x_train,
                        y_train,
                        epochs=40,
                        batch_size=512,
                        validation_data=(x_dev, y_dev),
                        verbose=1)
    history_dict = history.history
    history_dict.keys()
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
    plt.clf()  # clear figure
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
# def main():
#     x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
#     train(x_train, y_train, vocab_processor, x_dev, y_dev)
if __name__ == '__main__':
    # print(FLAGS.positive_data_file)
    # preprocess()
    train()
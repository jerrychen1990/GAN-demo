#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/14/19 4:37 PM
# @Author  : xiaowa

from model import GANModel
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import codecs
import pickle
from utils import show, anti_categorical


def get_sample(x, y, sample_rate):
    n = x.shape[0]
    tmp = np.random.random(n) <= sample_rate
    return x[tmp], y[tmp]


def pre_parse_data(x, y, cat_num, max_ptx, sample_rate=1.):
    x = x / max_ptx
    x = x.reshape(x.shape[0], -1)
    y = to_categorical(y, cat_num)
    x, y = get_sample(x, y, sample_rate)
    return x, y


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, y_train = pre_parse_data(X_train, y_train, 10, 255., 0.1)
    print("train data's info  X's shape:{0}, y's shape:{1}".format(X_train.shape, y_train.shape))

    cat_shape = (y_train.shape[1],)
    gen_shape = (X_train.shape[1],)

    gan_model = GANModel(gen_shape, cat_shape)
    gan_model.fit(X_train, y_train, 2)
    print("training finished")

    name = "gan_v1"
    path = "models/" + name
    print("save to {}".format(path))
    # with codecs.open(path, 'wb') as f:
    #     pickle.dump(gan_model, f)
    #
    # gan_model.save("gan_v1.pickle")
    gan_model.save(name)

    for num in range(10):
        print("drawing:{}".format(num))
        pic = gan_model.gen_number(num)
        show(pic)

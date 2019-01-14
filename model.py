#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/14/19 4:25 PM
# @Author  : xiaowa

from keras.models import Model
from keras.layers import Input, Dense, Dropout, dot
from keras.utils import to_categorical
import numpy as np
import random
import pickle
import codecs


def show_his(self, his):
    his = [h.history for h in his]
    his_list = [(k, [h[k][-1] for h in his]) for k in his[0].keys()]
    for metric, value_list in his_list:
        print(metric, value_list)


class GANModel:
    def __init__(self, gen_shape, cat_shape):
        self.cat_shape = cat_shape
        self.gen_shape = gen_shape
        self.g = self.__init_g()
        self.d = self.__init_d()
        self.gan = self.__init_gan()
        self.batch_size = 1000
        self.d_epochs = 2
        self.g_epochs = 2

        self.d_his = []
        self.g_his = []
        self.check_his = []

        print(u"model init finished")

    def __init_g(self):
        in_layer = Input(shape=self.cat_shape)
        x = Dense(512, activation="relu")(in_layer)
        x = Dense(1024, activation="relu")(x)
        out_layer = Dense(self.gen_shape[0], activation="sigmoid")(x)
        g = Model(in_layer, out_layer)
        return g

    def __init_d(self):
        x_in_layer = Input(shape=self.gen_shape)
        x = Dense(512, activation="relu")(x_in_layer)
        x = Dropout(0.2)(x)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(self.cat_shape[0], activation="softmax")(x)
        y_in_layer = Input(shape=self.cat_shape)
        out_layer = dot([x, y_in_layer], axes=1)
        d = Model([x_in_layer, y_in_layer], out_layer)
        d.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        return d

    def __init_gan(self):
        x_in_layer = Input(shape=self.cat_shape)
        x = self.g(x_in_layer)
        out_layer = self.d([x, x_in_layer])
        gan = Model(x_in_layer, out_layer)
        gan.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        return gan

    def save(self, name):
        path = "models/" + name
        self.g.save(path + "g.h5")
        self.d.save(path + "d.h5")

        #
        #
        # print("save to {}".format(path))
        # with codecs.open(path, 'wb') as f:
        #     pickle.dump(self, f)

    def random_check(self, m_y):
        idx = random.randint(0, m_y.shape[0])
        y = m_y[idx].reshape(1, -1)
        x = self.g.predict(y)
        self.check_his.append((y, x))

    def gen_number(self, n):
        y = to_categorical(n, self.cat_shape[0])
        rs = self.g.predict(y.reshape(1, -1))
        return rs

    @staticmethod
    def set_trainable(model, v):
        model.trainable = v
        for layer in model.layers:
            layer.trainable = v

    def fit(self, x, y, epochs, batch_size=1000):
        for idx in range(epochs):
            print("epochs:{}".format(idx + 1))
            print("predicting...")
            m_y = y
            self.random_check(m_y)

            m_x = self.g.predict(m_y)
            d_x = np.concatenate((x, m_x))
            d_y = np.concatenate((y, m_y))
            d_out = np.concatenate((np.ones(x.shape[0]), np.zeros(x.shape[0])))
            print("train discriminator...")
            GANModel.set_trainable(self.d, True)
            self.d_his.append(self.d.fit([d_x, d_y], d_out, batch_size=1000, epochs=self.d_epochs))

            print("train generator...")
            GANModel.set_trainable(self.d, False)

            gan_y = np.ones(x.shape[0])
            self.g_his.append(self.gan.fit(y, gan_y, batch_size=1000, epochs=self.g_epochs))

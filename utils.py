#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/14/19 6:59 PM
# @Author  : xiaowa
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt


def add_noise(arr, scale=0.1):
    noise = np.random.normal(size=arr.shape, scale=scale)
    return arr + noise


def show(sample, max_pt=255.):
    n = sqrt(sample.shape[1])
    sample = sample.reshape(-1, n)
    sample *= max_pt
    sample = sample.astype(int)
    plt.imshow(sample)
    plt.show()


def anti_categorical(arr):
    return np.array([np.where(r == 1)[0][0] for r in arr])

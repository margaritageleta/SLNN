"""Pattern recognition with Single Layer Neural Network

Set of functions to train and evaluate SLNNs using different
optimization methods. Intented for recognizing digits.

This file can also be used as a python module.

"""

import numpy as np
import matplotlib.pyplot as plt


# -- Image utils --

nums = np.array([
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
    [0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0]
])


def num_show(num):
    img_array = num.reshape((-1, 5))
    plt.imshow(img_array, cmap='gray')


# -- Data generation --


def add_noise(number, noise_freq):
    """
    Returns a copy of `number` with some percent
    of the pixels modified.
    """

    n_pixels = round(noise_freq * len(number))
    swap = np.random.randint(0, 35, n_pixels)
    num = number.copy()
    num[swap] = (num[swap] + 1) % 2
    return num


def gen_data(seed, train_size, num_target, tr_freq=0.5, noise_freq=0):
    """
    Returns both train and test data and their labels.

    Numbers for training data are chosen first from 
    either `num_target` or the full set of digits, and
    then selected uniformly.
    """

    train = np.empty((train_size, 35))
    test = np.empty((train_size*10, 35))
    y_tr = np.empty(train_size)
    y_te = np.empty(train_size*10)

    # Train data
    nrows_target = round(max(0.1*len(num_target), tr_freq) * train_size)
    for i in range(nrows_target):
        train[i] = add_noise(nums[num_target[i % len(num_target)]], noise_freq)
        y_tr[i] = num_target[i % len(num_target)]

    for i in range(nrows_target, train_size):
        train[i] = add_noise(nums[i % len(nums)], noise_freq)
        y_tr[i] = i % len(nums)

    # Test data
    for i in range(train_size*10):
        test[i] = add_noise(nums[i % len(nums)], noise_freq)
        y_te[i] = i % len(nums)

    np.random.seed(seed)
    tr_p = np.random.permutation(train_size)
    te_p = np.random.permutation(train_size*10)

    return train[tr_p], test[te_p], y_tr[tr_p], y_te[te_p]
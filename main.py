"""Pattern recognition with Single Layer Neural Network

Set of functions to train and evaluate SLNNs using different
optimization methods. Intented for recognizing digits.

This file can also be used as a python module.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize.linesearch import line_search


# -- Image utils --

nums = np.array([
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0], #0
    [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0], #1
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1], #2
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0], #3
    [0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], #4
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0], #5
    [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0], #6
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], #7
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0], #8
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0]  #9
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

    n_pixels = int(np.ceil(noise_freq * len(number)))
    swap = np.random.randint(0, 35, n_pixels)
    num = number.copy()
    num[swap] = (num[swap] + 1) % 2
    return num


def gen_data(seed, train_size, num_target, tr_freq, noise_freq):
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
    nrows_target = int(np.ceil(max(0.1*len(num_target), tr_freq) * train_size))
    for i in range(nrows_target):
        train[i] = add_noise(nums[num_target[i % len(num_target)]], noise_freq)
        y_tr[i] = 1

    not_target = [i for i in range(10) if i not in num_target]
    for i in range(nrows_target, train_size):
        train[i] = add_noise(nums[not_target[i % len(not_target)]], noise_freq)
        y_tr[i] = 0

    # Test data
    for i in range(train_size*10):
        test[i] = add_noise(nums[i % len(nums)], noise_freq)
        y_te[i] = 1 if i % len(nums) in num_target else 0

    np.random.seed(seed)
    tr_p = np.random.permutation(train_size)
    te_p = np.random.permutation(train_size*10)

    return train[tr_p], test[te_p], y_tr[tr_p, None], y_te[te_p, None]

# -- SLNN --


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def y(X, w):
    """Evaluates a SLNN with weigths `w` """

    return sigmoid((sigmoid(X) @ w.T))

# -- Objective functions --


def loss(w, X, ytr, p=0):
    #return np.sum((y(X, w) - ytr)**2) + p/2 * np.sum(w**2)
    return np.array([np.linalg.norm(y(X, w) - ytr)**2 + p/2 * np.linalg.norm(w)**2])


def g_loss(w, X, ytr, p=0):
    #return (2 * sigmoid(X) * ((y(X, w) - ytr) * y(X, w) * (1 - y(X, w))) + p*w).sum(axis=0)
    return np.squeeze(2 * sigmoid(X.T) @ ((y(X, w) - ytr) * y(X, w) * (1 - y(X, w))) + p*w.T)

# -- Optimization --


def GM(x, f, g, eps, kmax):
    k = 0
    while np.linalg.norm(g(x)) > eps and k < kmax:
        d = -g(x)
        alpha, *_ = line_search(f, g, x, d)
        x = x + alpha*d
        k += 1

    return x


def BFGS(x, f, g, eps, kmax):
    H = I = np.identity(len(g(x)))
    k = 0
    while np.linalg.norm(g(x)) > eps and k < kmax:
        d = -H @ g(x)
        alpha, *_ = line_search(f, g, x, d, c1 = 0.01, c2 = 0.45)
        print(alpha)
        if alpha is None:
            return x
        x, x_prev = x + alpha*d, x
        s = x - x_prev
        y = g(x) - g(x_prev)
        y = y[None, :]
        rho = 1 / ((y).T @ s)
        H = (I - rho * s @ y.T) @ H @ (I - rho * y @ (s.T)) + rho * s @ s.T
        k += 1
    return x


"""
 BLS params:
 ----------
         kmaxBLS = 30, 
         epsilonBLS = 1.0e-03, 
         c1 = 0.01, 
         c2 = 0.45
         
 optimizer:
 ---------
         Steepest descent (GM)
         Conjugate Gradient method (CGM)
         Quasi-Newton BFGS
""" 

# def nnet(Xtr, Ytr,lambda = 0.00, epsilon = 1.0e-06, kmax = 500, BLS_params, optimizer):
#     # init rand weights
#     w = rand_weights
    
#     if optimizer == "GM": # Steepest descent
#         objective_func = loss(w, Xtr, Ytr, p = lambda)
#         GM(x, f, , BLS_params, eps = epsilon, kmax = kmax)
#         return 0
#     ##
#     elif optimizer == "CGM": # Conjugate Gradient method
        
#         return 1
#     ##
#     elif optimizer == "BFGS": # Quasi-Newton BFGS
        
#         return 2
#     ##
#     else:
#         print("Invalid optimizer.")
        
    





"""Pattern recognition with Single Layer Neural Network

Set of functions to train and evaluate SLNNs using different
optimization methods. Intented for recognizing digits.

This file can also be used as a python module.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import line_search


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
    # return np.sum((y(X, w) - ytr)**2) + p/2 * np.sum(w**2)
    return np.linalg.norm(y(X, w) - ytr)**2 + p/2 * np.linalg.norm(w)**2


def g_loss(w, X, ytr, p=0):
    # return (2 * sigmoid(X) * ((y(X, w) - ytr) * y(X, w) * (1 - y(X, w))) + p*w).sum(axis=0)
    return np.squeeze(2 * sigmoid(X.T) @ ((y(X, w) - ytr) * y(X, w) * (1 - y(X, w))) + p*w.T)


# -- Optimization --

def GM(x, f, g, eps, kmax, precision=6):
    gradient_norm = np.round(np.linalg.norm(g(x)), precision)
    Xk = [[np.NaN, f(x), gradient_norm]]
    rk = [np.NaN]
    Mk = [np.NaN]
    # ============== #
    k = 0
    while np.linalg.norm(g(x)) > eps and k < kmax:
        d = -g(x)
        if k > 0:
            alpha, *_ = line_search(f, g, x, d, old_old_fval=f(x_prev), c1=0.01, c2=0.45)
        else:
            alpha, *_ = line_search(f, g, x, d, c1=0.01, c2=0.45)
        if alpha is None:
            print("alpha not found (!)")
            break
        x, x_prev = x + alpha*d, x
        k += 1
        # =========== #
        gradient_norm = np.round(np.linalg.norm(g(x)), precision)
        Xk.append([alpha, f(x), gradient_norm])
        rk.append(np.linalg.norm(g(x))/np.linalg.norm((g(x_prev))))
        Mk.append(np.linalg.norm(g(x))/(np.linalg.norm((g(x_prev)))**2))
        # =========== #
    data = pd.DataFrame(Xk, columns=["alpha", "f(x)", "||g(x)||"], dtype=np.float)
    data['r'] = rk
    data['M'] = Mk
    return x, data


def CGM(x, f, g, eps, kmax, iCG, iRC, nu=None, precision=6):
    gradient_norm = round(np.linalg.norm(g(x)), precision)
    Xk = [[np.NaN, f(x), gradient_norm]]
    rk = [np.NaN]
    Mk = [np.NaN]
    # ============== #
    d = -g(x)
    k = 0
    while np.linalg.norm(g(x)) > eps and k < kmax:
        if k > 0:
            alpha, *_ = line_search(f, g, x, d, old_old_fval=f(x_prev), c1=0.01, c2=0.45)
        else:
            alpha, *_ = line_search(f, g, x, d, c1=0.01, c2=0.45)
        if alpha is None:
            break
        x, x_prev = x + alpha*d, x
        # =========== #
        # CGM variants
        if iCG == "FR":
            beta = (g(x).T @ g(x)) / (g(x_prev).T @ g(x_prev))
        elif iCG == "PR":
            beta = max(0, g(x).T @ (g(x) - g(x_prev)) / (g(x_prev).T @ g(x_prev)))
        else:
            raise TypeError("iCG should be FR (Fletcher-Reeves) or PR (Polak-Ribière)")
        # Restart conditions
        if iRC > 0 and nu is None:
            raise TypeError(f"nu is a necessary parameter with iRC equal to {iRC}")
        if (iRC == 1 and k % nu == 0 or
                iRC == 2 and g(x).T @ g(x_prev) / np.linalg.norm(g(x))**2 > nu or
                k == 0):
            d = -g(x)
        else:
            d = -g(x) + beta*d
        k += 1
        # =========== #
        gradient_norm = np.round(np.linalg.norm(g(x)), precision)
        Xk.append([alpha, f(x), gradient_norm])
        rk.append(np.linalg.norm(g(x))/np.linalg.norm((g(x_prev))))
        Mk.append(np.linalg.norm(g(x))/(np.linalg.norm((g(x_prev)))**2))
        # =========== #
    data = pd.DataFrame(Xk, columns=["alpha", "f(x)", "||g(x)||"], dtype=np.float)
    data['r'] = rk
    data['M'] = Mk
    return x, data


def BFGS(x, f, g, eps, kmax, precision=6):
    gradient_norm = np.round(np.linalg.norm(g(x)), precision)
    Xk = [[np.NaN, f(x), gradient_norm]]
    rk = [np.NaN]
    Mk = [np.NaN]
    # =========== #
    H = I = np.identity(len(g(x)))
    k = 0
    while np.linalg.norm(g(x)) > eps and k < kmax:
        d = -H @ g(x)
        if k > 0:
            alpha, *_ = line_search(f, g, x, d, old_old_fval=f(x_prev), c1=0.01, c2=0.45)
        else:
            alpha, *_ = line_search(f, g, x, d, c1=0.01, c2=0.45)
        if alpha is None:
            break
        x, x_prev = x + alpha*d, x
        s = x - x_prev
        y = g(x) - g(x_prev)
        y = y[None, :]
        rho = 1 / ((y).T @ s)
        H = (I - rho * s @ y.T) @ H @ (I - rho * y @ (s.T)) + rho * s @ s.T
        k += 1
        # =========== #
        gradient_norm = np.round(np.linalg.norm(g(x)), precision)
        Xk.append([alpha, f(x), gradient_norm])
        rk.append(np.linalg.norm(g(x))/np.linalg.norm((g(x_prev))))
        Mk.append(np.linalg.norm(g(x))/(np.linalg.norm((g(x_prev)))**2))
        # =========== #
    data = pd.DataFrame(Xk, columns=["alpha", "f(x)", "||g(x)||"], dtype=np.float)
    data['r'] = rk
    data['M'] = Mk
    return x, data

# -- SLNN class --


class SLNN:
    """
    Single layer neural network, with n inputs.
    Methods:
        train:      minimize the loss function for some given train set and its
                    labels. The available optimizer methods are "GM", "CGM" and
                    "BFGS".
        cvtrain:    uses cross-validation to find the best regularization
                    parameter, and trains the SLNN with it.
        predict:    predicts the labels for some given data.
        accuracy:   given some data set and its labels, computes the accuracy
                    of the model.
        summary:    prints a summary of the model and returns the iterations.
    """

    def __init__(self, n=35):
        self.weights = np.zeros((1, 35))
        self._trained = False
        self.out = None

    def train(self, optimizer, x, y, p=0, epsilon=10e-6, kmax=1000):
        if optimizer == "GM":
            self.weights, self.out = GM(self.weights, lambda w: loss(w, x, y, p), lambda w: g_loss(w, x, y, p), epsilon, kmax)
        elif optimizer == "CGM":
            self.weights, self.out =  CGM(self.weights, lambda w: loss(w, x, y, p), lambda w: g_loss(w, x, y, p), epsilon, kmax, "FR", 0)
        elif optimizer == "BFGS":
            self.weights, self.out = BFGS(self.weights, lambda w: loss(w, x, y, p), lambda w: g_loss(w, x, y, p), epsilon, kmax)
        else:
            raise ValueError("Invalid optimizer.")
        self.x, self.y, self.p = x, y, p
        self.optimizer = optimizer
        self._trained = True

    def cvtrain(self, optimizer, x, y, epsilon=10e-6, kmax=1000):
        Xtr = x[len(x)//5:]
        ytr = y[len(y)//5:]
        Xval = x[:len(x)//5]
        yval = y[:len(y)//5]

        best_lambda = 0
        best_acc = 0
        for p in np.arange(0, 3, 0.3):
            self.weights = np.zeros((1, len(self.weights[0])))
            self.train(optimizer, Xtr, ytr, p, epsilon, kmax)
            if self.accuracy(Xval, yval) > best_acc:
                best_acc = self.accuracy(Xval, yval)
                best_lambda = p
        self.train(optimizer, x, y, best_lambda, epsilon, kmax)

    def predict(self, x):
        return np.round(y(x, self.weights))

    def accuracy(self, x, y):
        return 100 * np.sum(self.predict(x) == y) / len(y)

    def summary(self, Xte, yte):
        if not self._trained:
            print("Model is not trained yet")
        else:
            print(f"Single Layer Neural Network (SLNN)")
            print("-"*75)
            print(f"Input size:\t\t\t{len(np.squeeze(self.weights))}")
            print(f"Output size:\t\t\t1 (binary)")
            print("-"*75)
            print(f"Train data:\t\t\t{len(self.x)} observations")
            print(f"Chosen optimization routine:\t{self.optimizer}")
            print(f"Regularization parameter:\t{self.p}")
            print(f"Loss:\t\t\t\t{loss(self.weights, self.x, self.y, self.p)}")
            print(f"Training accuracy:\t\t{self.accuracy(self.x, self.y)}%")
            print(f"Test accuracy:\t\t\t{self.accuracy(Xte, yte)}%")
            print("-"*75)
            print("Gradient: \n", g_loss(self.weights, self.x, self.y, self.p))
            print("-"*75)
            num_show(self.weights)
            return self.out

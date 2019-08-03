import numpy as np


def linesearch(f, g0, x0, d, alpham, c1, c2, maxiter, eps):

    alpha0 = 0
    alphap = alpha0

    def g(x):
        return g0(x).T

    iout = 0
    if c1 == 0:
        c1 = 1e-4
    if c2 == 0:
        c2 = 0.5

    alphax = alpham
    fx0 = f(x0)
    gx0 = g(x0) @ d
    fxp = fx0
    gxp = gx0

    i = 1
    while (i < maxiter):
        if np.abs(alphap - alphax) < eps:
            iout = 2
            alphas = alphax
            return alphas, iout
        xx = x0 + alphax*d
        fxx = f(xx)
        gxx = g(xx) @ d

        if (fxx > fx0 + c1*alphax*gx0) or (i > 1 and fxx >= fxp):  # WC1
            alphas, iout_zoom = zoom(f, g, x0, d, alphap, alphax, c1, c2, eps)
            if iout_zoom == 2:
                iout = 2
            return alphas, iout
        if np.abs(gxx) <= -c2 * gx0:  # SWC
            alphas = alphax
            return alphas, iout
        if gxx >= 0:
            alphas, iout_zoom = zoom(f, g, x0, d, alphax, alphap, c1, c2, eps)
            if iout_zoom == 2:
                iout = 2
            return alphas, iout
        alphap = alphax
        fxp = fxx
        gxp = gxx
        alphax = alphax + (alpham - alphax)*np.random.rand(1)[0]
        i = i+1

    if i == maxiter:
        iout = 1
        alphas = alphax
    return alphas, iout


def zoom(f, g, x0, d, alphal, alphah, c1, c2, eps):
    fx0 = f(x0)
    gx0 = g(x0) @ d
    iout = 0

    while (True):
        alphax = 1/2 * (alphal + alphah)
        if np.abs(alphal - alphah) < eps:
            iout = 2
            alphas = alphax
            return alphas, iout
        xx = x0 + alphax*d
        fxx = f(xx)
        gxx = g(xx) @ d
        xl = x0 + alphal*d
        fxl = f(xl)
        if (fxx > fx0 + c1*alphax*gx0) or (fxx >= fxl):
            alphah = alphax
        else:
            if np.abs(gxx) <= -c2*gx0:
                alphas = alphax
                return alphas, iout
            if gxx*(alphah - alphal) >= 0:
                alphah = alphal
            alphal = alphax
    return alphas, iout


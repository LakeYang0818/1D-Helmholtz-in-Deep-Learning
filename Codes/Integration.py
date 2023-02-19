from scipy.special import roots_jacobi, jacobi, gamma
import numpy as np
import torch

def gauss_lobatto_jacobi_weights(q, alpha = 0, beta = 0):
    w = []
    x = roots_jacobi(q - 2, alpha + 1, beta + 1)[0]

    if alpha == 0 and beta == 0:
        # weight at x
        w  = 2 / ((q-1) * q * jacobi(q - 1, 0, 0)(np.array(x )) ** 2)
        # weight at xmin
        wl = 2 / ((q-1) * q * jacobi(q - 1, 0, 0)(np.array(-1)) ** 2)
        # weight at xmax
        wr = 2 / ((q-1) * q * jacobi(q - 1, 0, 0)(np.array(+1)) ** 2)
    else:
        # weight at x
        w  = 2 ** (alpha + beta + 1) * gamma(alpha + q) * gamma(beta + q) \
             / ((q - 1) * gamma(q) * gamma(alpha + beta + q + 1) * (jacobi(q - 1, alpha, beta)(np.array(x )) ** 2))
        # weight at xmin
        wl = 2 ** (alpha + beta + 1) * gamma(alpha + q) * gamma(beta + q) * (beta + 1) \
             / ((q - 1) * gamma(q) * gamma(alpha + beta + q + 1) * (jacobi(q - 1, alpha, beta)(np.array(-1)) ** 2))
        # weight at xmax
        wr = 2 ** (alpha + beta + 1) * gamma(alpha + q) * gamma(beta + q) * (alpha + 1) \
             / ((q - 1) * gamma(q) * gamma(alpha + beta + q + 1) * (jacobi(q - 1, alpha, beta)(np.array(+1)) ** 2))

    w = np.append(w, wr) # write weight of xmax on the right of weight of x
    w = np.append(wl, w) # write weight of xmin on the left  of weight of x
    x = np.append(x, 1) # write xmax on the right of x
    x = np.append(0, x) # write xmin on the left of x
    return [x, w]

def gauss_lobatto_jacobi_quadrature1D(num_points, a, b, alpha = 0, beta = 0):
    roots, weights = gauss_lobatto_jacobi_weights(num_points, alpha, beta)
    X = (b - a) / 2 * (roots + 1) + a
    X = torch.tensor(X)
    return X, torch.tensor(weights)

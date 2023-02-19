from typing import Any
import numpy as np
from scipy.special import binom, chebyt, gamma, jacobi
import torch

"""
The d-th derivative of a one-dimensional test function, evaluated at a single point.
:param x: the point at which to evaluate the test function.
:param index: the index of the test function
:param d: the order of the derivative
:param type: the type of the test function
:return: the test function evaluated at that point
"""

def _chebyshev(x: Any, n: int, *, d: int) -> float:
    n = n.item() if isinstance(n, np.ndarray) else n
    if d > n:
        return torch.zeros_like(x)
    elif d == 0:
        res = chebyt(n)(x)
        return res
    else:
        res = 0
        for i in range(n - d + 1):
            if (n - d) % 2 != i % 2:
                continue
            A = binom((n + d - i) / 2 - 1, (n - d - i) / 2)
            B = gamma((n + d + i) / 2)
            C = gamma((n - d + i) / 2 + 1)
            D = _chebyshev(x, i, d=0)
            v = A * B / C * D
            if i == 0:
                v *= 1.0 / 2
            res += v
        return torch.tensor(2 ** d * n * res)

def _jacobi(x: Any, n: int, *, d: int, a: int, b: int) -> float:
    if d == 0:
        return jacobi(n, a, b)(x)
    elif d > n:
        return torch.zeros_like(x)
    else:
        res = (gamma(a + b + n + 1 + d)
               / (2 ** d * gamma(a + b + n + 1))
               * jacobi(n - d, a + d, b + d)(x))
        return res

def call_tf(x, index, d, typetest):
    x = torch.tensor(x)
    if typetest == "chebyshev":
        return _chebyshev(x, index + 1, d=d) - _chebyshev(x, index - 1, d=d)
    elif typetest == "legendre":
        return _jacobi(x, index + 1, a=0, b=0, d=d) - _jacobi(x, index - 1, a=0, b=0, d=d)
    else:
        pass

"""
N = 3
for n in range(1, N+1):
    x = torch.tensor([1., 2., 4.])
    print(call_tf(x=x, index=n, d=0, typetest="legendre"))"""

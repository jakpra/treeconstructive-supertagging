'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

import math
from copy import deepcopy


def _bottom_up(i, j, l):
    return (j - i) * 10**(math.floor(math.log(l, 10)) + 1) + i


def bottom_up(l):
    return lambda i, j: _bottom_up(i, j, l)


def _top_down(i, j, l):
    return (i - j) * 10**(math.floor(math.log(l, 10)) + 1) + i


def top_down(l):
    return lambda i, j: _top_down(i, j, l)


def _left_corner(i, j, l):
    return j * 10**(math.floor(math.log(l, 10)) + 1) + (j - i)


def left_corner(l):
    return lambda i, j: _left_corner(i, j, l)


def pre_order(a):
    return str(a)


def harmonic_mean(a, b):
    return ((2 * a * b) / (a + b)) if a + b > 0 else 0.


def sparse_copy_and_update(orig, update):
    shallow_copy = orig.copy()
    for k, v in update.items():
        if isinstance(v, dict):
            shallow_copy[k] = sparse_copy_and_update(orig.get(k, {}), v)
        else:
            shallow_copy[k] = deepcopy(v)

    return shallow_copy


def binary_to_decimal(binary):
    decimal, i, n = 0, 0, 0
    while (binary != 0):
        dec = binary % 10
        decimal = decimal + dec * pow(2, i)
        binary = binary // 10
        i += 1
    return decimal


if __name__ == '__main__':
    d1 = {'a':{'b': {'c':1}}, 'b':{'b': {'c':1}, 'c':{'d':3}}}
    d2 = {'b':{'b':{'d':2}}}
    d3 = {'b': {'b': {'c': 2}}}
    # d4 = {'b': {'b': {'d': 2}}}

    print('d1', d1)

    d1_2 = sparse_copy_and_update(d1, d2)
    print('d1', d1)
    print('d2', d2)
    print('d1_2', d1_2)

    d1_3 = sparse_copy_and_update(d1, d3)
    print('d1', d1)
    print('d3', d3)
    print('d1_3', d1_3)

    print(binary_to_decimal(1000))

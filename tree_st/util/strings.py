'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

def open_par(s):
    return s == '('


def close_par(s):
    return s == ')'


def open_angle(s):
    return s == '<'


def close_angle(s):
    return s == '>'


def open_bracket(s):
    return s == '['


def close_bracket(s):
    return s == ']'


def slash(s):
    return s in ('/', '\\')


def one_bit(s):
    return s == '1'


def zero_bit(s):
    return s == '0'

def tree_node(s):
    return s == 'T'

def leaf_node(s):
    return s == 'L'

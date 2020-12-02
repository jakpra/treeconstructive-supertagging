'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

from .derivation import Derivation
from .category import Category


class Dependency:
    def __init__(self, dep, head, head_cat: Category, arg, *args, derivation: Derivation = None, **kwargs):
        self.dep = dep
        self.head = head
        self.head_cat = head_cat
        self.arg = arg
        self.head_args = self.head_cat.nargs()
        self.dep_args = None
        self.derivation = None
        if derivation is not None:
            self.set_derivation(derivation)

    def get_arg_addr(self):
        return 2**(self.head_args - self.arg + 1) + 1

    def get_head_addr(self):
        return 2**(self.dep_args - self.get_arg().nargs())

    def get_arg(self):
        return self.head_cat.get_category(self.get_arg_addr(), binary=False)

    def get_dep(self):
        return None if self.derivation is None else self.derivation.get_node(self.dep, self.dep+1)

    def get_head(self):
        return None if self.derivation is None else self.derivation.get_node(self.head, self.head+1)

    def set_derivation(self, derivation):
        derivation.dependencies[self.head][self.get_arg_addr()] = self
        self.derivation = derivation
        self.dep_args = self.get_dep().category1.nargs()

    def __str__(self):
        return f'({self.head}, {self.get_head()}) -({self.arg}, {self.get_arg()})-> ({self.dep}, {self.get_dep()})'


class Dependencies:
    pass

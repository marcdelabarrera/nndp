from dataclasses import dataclass
from typing import Callable

import jax
from jax import Array
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@dataclass
@register_pytree_node_class
class Policy:
    params: dict
    nn: Callable[[Array,dict], Array]

    def tree_flatten(self):
        return (self.params, ), self.nn
    
    @classmethod
    def tree_unflatten(cls, nn, params):
        return cls(*params, nn)

    def __call__(self, state:Array)->Array:
        state = jnp.atleast_2d(state)
        return self.nn(state, self.params)
    
    def update(self, params:dict)->None:
        self.params = params
        return self

    @property
    def structure(self):
        return jax.tree_map(lambda x: x.size, self.params)
    
    @property
    def size(self):
        '''
        Returns the number of parameters
        '''
        return jax.tree_util.tree_reduce(lambda x, y: x + y, self.structure)


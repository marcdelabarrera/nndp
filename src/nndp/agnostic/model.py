from dataclasses import dataclass
from typing import Callable

from jax import Array


@dataclass(frozen=True)
class Model:
    '''
    Class that defines a model
    '''
    u: Callable[[Array, Array], Array]
    m: Callable[[Array, Array, Array], Array] 
    T:int
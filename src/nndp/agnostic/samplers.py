# Standard imports
import time
from dataclasses import dataclass

# Third party imports

import jax.numpy as jnp
from jax import Array
import jax

@dataclass
class Sampler:
    k: Array
    x0: Array

    def __post_init__(self):
        if self.k.shape[0] != self.x0.shape[0]:
            raise ValueError('k and x0 must have the same number of rows')

    def initial_state(self, n:int=1, seed=None)->Array:
        return self.state(t=0, n=n, seed = seed)
    

    def state(self, t:int=None, n:int=1, seed=None)->Array:
        seed = seed if seed else int(time.time()*1e7)
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        if t is None:
            t = jax.random.randint(subkey, 0, minval=0, maxval=self.k.shape[1])
            key, subkey = jax.random.split(key)
        k0 = jax.random.choice(subkey, self.k[:,t], (n,), replace=True)
        key, subkey = jax.random.split(key)
        x0 = jax.random.choice(subkey, self.x0, (n,), replace=True)
        return jnp.column_stack([k0,x0])


    def path(self, initial_state: Array=None, n:int=1, seed=None)->tuple[Array, Array]:
        '''
        Generates n paths, all starting from the same initial state.
        #TODO: different if n=1
        '''
        seed = seed if seed else int(time.time()*1e7)
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        if initial_state is None:
            initial_state = self.initial_state(n=1, seed = subkey[0])[0]
            key, subkey = jax.random.split(key)
        k = self.k[(self.k[:,0]== initial_state[:self.k.shape[2]]).all(axis=1)]
        k = jax.random.choice(subkey, k, (n,), replace=True)
        initial_state = jnp.tile(initial_state, (n,1))
        x0 = initial_state[:,-self.x0.shape[1]:]
        return x0, k

       
  



# Standard imports
from typing import Callable

# Third party imports
import jax 
import jax.numpy as jnp
from jax import Array
from jax.tree_util import Partial

# Economic parameters used in functions
T = 10 # number of periods: t=0, ...,T where death occurs at T
w = 1
beta = 0.98 # discount factor
alpha = 0.6
kappa_h = 1

@jax.jit
def u(state:Array, action:Array) -> Array:
    '''
    Reward function
    '''
    t, z, n_ = state[...,0], state[...,1], state[...,2]
    n = action[...,0]
    return (beta**t)*(z*n**alpha-w*n-kappa_h*(n-n_)**2)

@jax.jit
def m(key:jax.random.PRNGKey, state:Array, action:Array) -> Array:
    '''
    State evolution equation
    '''
    t, z, n_ = state[...,0], state[...,1], state[...,2]
    n = action[...,0]
    t_next = t + 1
    y_next = jnp.exp(rho * jnp.log(y) + sigma_y * jax.random.normal(key, shape = (N,1)))
    n_next = R * (a + y - c)
    return jnp.column_stack([t_next, y_next, a_next])

@jax.jit
def Gamma(state:Array) -> list[tuple[Array,Array]]:
    '''
    Define bounds of action in each state
    '''
    return [(jnp.ones((state.shape[0],1))*1e-6, state[:,[1]]+state[:,[2]])]

@Partial(jax.jit,static_argnames='N')
def F(key:jax.random.PRNGKey, N:int) -> Array:
    '''
    Sample N initial states
    '''
    t = jnp.zeros(N)
    key, subkey = jax.random.split(key)
    y = jnp.exp(jax.random.uniform(subkey, shape = (N,1), minval = logy_bound[0], maxval = logy_bound[1]))
    key, subkey = jax.random.split(key)
    a = jax.random.uniform(subkey, shape = (N,1), minval = a_bound[0], maxval = a_bound[1])
    state = jnp.column_stack([t, y, a])
    return state

@Partial(jax.jit, static_argnames=['nn'])
def policy(state:Array, 
           params:dict,
           nn:Callable[[dict, Array], Array]
           ) -> Array:
    '''
    Defines how a Haiku Neural Network, nn, with parameters, params, is mapped
    into an action.
    
    Parameters:
    -----------
    state: current state = N_simul x n_states
    nn: Haiku Neural Network with signature nn(params, state)
    params: dictionary of parameters used by nn. 

    Returns:
    -----------
    action: action to take = N_simul x n_actions.
    '''
    c_min, c_max = Gamma(state)[0]
    action = c_min + nn(params, state) * c_max
    return action

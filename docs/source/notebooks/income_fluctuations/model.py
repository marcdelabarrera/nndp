# Standard imports
from typing import Callable

# Third party imports
import jax 
import jax.numpy as jnp
from jax import Array
from jax.tree_util import Partial

# Economic parameters used in functions
T = 10 # number of periods: t=0, ...,T where death occurs at T
beta = 0.98 # discount factor
gamma = 1.1 # relative risk aversion
R = 1/beta # interest rate
rho = 0.9 # persistence of AR(1)
sigma_y = 0.2 # transitory shock variable
sigma_y_st = sigma_y / jnp.sqrt(1 - rho**2) # stationary variable of AR(1)
logy_bound = [-2 * sigma_y_st, 2 * sigma_y_st] # bound for log income
a_bound = [0, 10] # bound for assets

@jax.jit
def u(state:Array, action:Array) -> Array:
    '''
    Reward function
    '''
    t = state[...,0]
    c = action[...,0]
    return jnp.where(t<=T, (beta**t)*(c ** (1. - gamma))/(1. - gamma),0)

@jax.jit
def m(key:jax.random.PRNGKey, state:Array, action:Array) -> Array:
    '''
    State evolution equation
    '''
    N = state.shape[0]
    t, y, a = state[...,0], state[...,1], state[...,2]
    c = action[...,0]
    t_next = t + 1
    y_next = jnp.exp(rho * jnp.log(y) + sigma_y * jax.random.normal(key, shape = (N,)))
    a_next = R * (a + y - c)
    return jnp.column_stack([t_next, y_next, a_next])

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
    y, a = state[...,1], state[...,2]
    return nn(params, state) * (y+a).reshape(-1,1)

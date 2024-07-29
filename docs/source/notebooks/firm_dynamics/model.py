


# Standard imports
from typing import Callable

# Third party imports
import jax 
import jax.numpy as jnp
from jax import Array
from jax.tree_util import Partial

# Economic parameters used in functions
T = 10 # number of periods: t=0, ...,T where death occurs at T
r = 0.04 # discount factor
alpha = 1/3
delta = 0.1

rho_z = 1
sigma_z = 0

@jax.jit
def u(state:Array, action:Array) -> Array:
    '''
    Reward function
    '''
    t, z, k = state[...,0], state[...,1], state[...,2]
    k_next = action[...,0]

    return ((1/(1+r))**t)*(jnp.exp(z)*k**alpha - (k_next-(1-delta)*k))



@jax.jit
def m(key:jax.random.PRNGKey, state:Array, action:Array) -> Array:
    '''
    State evolution equation
    '''
   
    t, z = state[...,0], jnp.atleast_1d(state[...,1])
    k_next = action[...,0]
    t_next = t + 1
    z_next = rho_z * z + sigma_z * jax.random.normal(key, shape = (len(z),))
    return jnp.column_stack([t_next, z_next, k_next])


@Partial(jax.jit, static_argnames='N')
def F(key:jax.random.PRNGKey, N:int) -> Array:
    '''
    Sample N initial states
    '''
    key, *subkey = jax.random.split(key, 4)
    t = jax.random.randint(subkey[0], shape=(N,), minval= 0, maxval = T)
    z = jax.random.uniform(subkey[1], shape = (N,), minval = -0.6, maxval = 0.6)
    k = jax.random.uniform(subkey[2], shape = (N,), minval = 0, maxval = 15)
    return jnp.column_stack([t, z, k])

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
    state = jnp.atleast_2d(state)
    #t, z, k = state[...,0], state[...,1], state[...,2]
    #t = t/T
    #z = z
    #k = (k-10)/10
    #state = jnp.column_stack([t, z, k])
    return nn(params, state)


def k_star(z:Array) -> Array:
    '''
    Computes the steady state capital level given z assuming infinite horizon
    '''
    return ((delta+r)/(alpha*jnp.exp(rho_z*z+1/2*sigma_z**2)))**(1/(alpha-1))
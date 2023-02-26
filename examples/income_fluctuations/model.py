from typing import Callable
import jax 
import jax.numpy as jnp
from jax._src.basearray import Array
from jax.tree_util import Partial
from jax._src.prng import PRNGKeyArray

# Economic parameters
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
def u(state:Array, 
      action:Array, 
      beta:float,
      gamma:float
      ) -> Array:
    '''
    utility function: Given a state and action, returns utility discounted
    back to time zero
    '''
    return (beta**state[:,[0]])*(action ** (1. - gamma))/(1. - gamma)

@jax.jit
def m(key:PRNGKeyArray,
      state:Array,
      action:Array,
      R:float,
      rho:float,
      sigma_y:float
      ) -> Array:
    '''
    Given a state and action, returns the next state assuming income
    follows an AR1
    '''
    N = state.shape[0]
    t, y, a = state[:,[0]], state[:,[1]], state[:,[2]]
    c = action
    t_next = t + 1
    y_next = jnp.exp(rho * jnp.log(y) + sigma_y * jax.random.normal(key, shape = (N,1)))
    a_next = R * (a + y - c)
    return jnp.column_stack([t_next, y_next, a_next])

@jax.jit
def Gamma(state:Array, A_min:int) -> list[tuple[Array,Array]]:
    '''
    Returns the feasible set of actions. c has to be positive and below a+y.
    Returns
    -------
    list of tuples. Lenght of the list is number of actions, 
    each tuple is the minimum value and maximum value.
    '''
    return [(jnp.ones((state.shape[0],1))*1e-6, state[:,[1]]+state[:,[2]]+A_min)]

@Partial(jax.jit,static_argnames='N')
def F(key:PRNGKeyArray, 
      N:int,
      a_min:float,
      a_max:float,
      logy_min:float, 
      logy_max:float) -> Array:
    '''
    Function to sample N initial states
    '''
    t = jnp.zeros(N)
    key, subkey = jax.random.split(key)
    y = jnp.exp(jax.random.uniform(subkey, shape = (N,1), minval = logy_min, maxval = logy_max))
    key, subkey = jax.random.split(key)
    a = jax.random.uniform(subkey, shape = (N,1), minval = a_min, maxval = a_max)
    state = jnp.column_stack([t, y, a])
    return state

@jax.jit
def policy(state:Array, 
           params:dict,
           nn:Callable[[dict, Array], Array], 
           Gamma:Callable[[Array], Array]           
           ) -> Array:
    '''
    Defines policy function that maps states to actions, depending
    on the bounds for each action.
    
    Parameters:
    -----------
    state: current state = N_simul x n_states
    nn: JAX Neural Network created with initialize_nn() that takes NN
        parameters and state as inputs
    Gamma: function that returns that upper and lower limit on each action
    params: dictionary of parameters used by nn. 

    Returns:
    -----------
    action: action to take = N_simul x n_actions.
    '''
    c_min, c_max = Gamma(state)[0]
    action = c_min + nn(params, state) * c_max
    return action


# Parameterize model
u = Partial(u, beta = beta, gamma = gamma)
m = Partial(m, R = R, rho = rho, sigma_y = sigma_y)
Gamma = Partial(Gamma, A_min = 0)
F = Partial(F, 
            a_min = a_bound[0], a_max = a_bound[1], 
            logy_min = logy_bound[0], logy_max = logy_bound[1]
            )
from dataclasses import dataclass

import matplotlib.pyplot as plt
import jax
from jax._src.prng import PRNGKeyArray
from jax._src.basearray import Array
from typing import Callable
import jax.numpy as jnp
from jax.tree_util import Partial
import optax



@Partial(jax.jit, static_argnames=['policy','nn','u','m', 'N_simul', 'T'])
def evaluate_policy(key:PRNGKeyArray,
                    policy: Callable[[Array, Array], Array],
                    params: dict,
                    nn: Callable,
                    u: Callable[[Array, Array], Array],
                    m: Callable[[PRNGKeyArray, Array, Array], Array], 
                    s0: Array,
                    T: int,
                    N_simul: int=1
                    ) -> Array:
    '''
    Calculates value function for a given set of initial states, simulating
    N_simul paths for each k=1,...,K initial states.
    
    Parameters:
    ----------
    key: JAX random number key
    policy: policy function
    u: instantaneous utility function
    m: law of motion
    s0: K x n_states array, where K is different initial states to consider
    T: number of periods
    N_simul: number of simulations to be considered to compute expectation
    
    Returns:
    ----------
    Vector of value functions at s0(k) = 
        E(\sum_{t=0}^T \beta^t u(action_t, state_t) | state_0 = s0(k))
    '''

    state = jnp.repeat(s0, repeats = N_simul, axis = 0)
    V = jnp.zeros(s0.shape[0] * N_simul)
    key, *subkey = jax.random.split(key, max((T + 1),2)) # ensure that works when T = 0

    V = jax.lax.fori_loop(0, T+1, 
                    body_fun = Partial(time_iteration, key=jnp.array(subkey), policy = policy,params=params,
                                       nn=nn, u=u,m=m), 
                    init_val = (V,state))[0].reshape(s0.shape[0], N_simul)
    return jnp.mean(V, axis = 1, keepdims = True)                  

def time_iteration(t,
                   x:tuple[Array, Array],
                   key:PRNGKeyArray,
                   policy,
                   params,
                   nn,
                   u,
                   m):
    V, state = x
    action = policy(state, params, nn)
    V += u(state, action)
    state_next = m(key[t], state, action)
    return V, state_next


def train_step(key:PRNGKeyArray,
               params:dict,
               policy:Callable[[Array, dict, Callable], Array],
               nn: Callable[[Array], Array],
               u:Callable[[Array, Array], Array],
               m:Callable[[PRNGKeyArray, Array, Array], Array], 
               s0:Array,
               T:int,
               N_simul:int,
               optimizer,
               opt_state
               ):
    '''
    Performs one optimization step to update neural network parameters, params,
    where objective function is -1 times the equal-weighted mean of value
    functions across K different initial states.
    
    Parameters:
    ----------
    key: JAX random number key
    params: dictionary of neural network parameters
    u: Instantaneous utility function
    m: Law of motion
    s0: Initial state = K x n_states array, where K is different
        initial states to consider
    T: number of periods
    N_simul: number of simulations to compute value function for each s0
    optimizer: optax optimizer object
    opt_state: state of optax optimizer
    
    Returns:
    ----------
    params: updated neutral network parameters
    opt_state: updated state of optimizer
    value: objective function value at previous step
    '''
    value, grads = jax.value_and_grad(lambda params: -jnp.mean(
        evaluate_policy(key = key, 
                       policy = policy,
                       params = params, 
                        nn= nn,
                       u = u,
                       m = m,
                       s0 = s0,
                       T = T,
                       N_simul = N_simul)
        )
    )(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, value



@dataclass
class TrainResult:
    values:Array

    def plot_convergence(self, ax = None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.values, **kwargs)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Objective Function')
        ax.set_title('Convergence of Objective Function')
        return ax
    

def train(key:PRNGKeyArray,
          params:dict,
          policy:Callable[[Array, dict, Callable], Array],
          nn: Callable[[Array], Array],
          u:Callable[[Array, Array], Array],
          m:Callable[[PRNGKeyArray, Array, Array], Array], 
          F:Callable[[Array, Array], Array],
          T:int,
          N_simul:int,
          batch_size:int,
          epochs:int,
          optimizer
          ):
    '''
    Optimizes parameters of a neural network, policy, by performing multiple steps
    of stochastic gradient descent. 
    
    Parameters:
    ----------
    key: JAX random number key
    params: dictionary of initialized neural network parameters
    policy: neural network to optimize
    u: Instantaneous utility function
    m: Law of motion
    F: function to initialize states
    T: number of periods
    N_simul: number of simulations to compute value function for each s0
    batch_size: number of initial states, s0, to sample in each optimization step
    N_iter: number of optimization steps
    optimizer: optax optimizer object
    
    Returns:
    ----------
    params: neural network parameters at final step of optimization
    '''
    opt_state = optimizer.init(params)
    values = jnp.empty(epochs)

    for i in range(epochs):
        key, *subkey = jax.random.split(key,3)
        s0 = F(key = subkey[0], N = batch_size)
        params, opt_state, value = train_step(key = subkey[1], 
                                              params = params, 
                                              policy = policy, 
                                              nn = nn,
                                              u = u, 
                                              m = m, 
                                              s0 = s0, 
                                              T = T, 
                                              N_simul = N_simul, 
                                              optimizer = optimizer, 
                                              opt_state = opt_state)
        
        values = values.at[i].set(value)

        print(f'\rObjective value on training iteration {i} out of {epochs}: {-value}', end='')
    return params, TrainResult(-values)


def simulate(key:Array,
             policy:Callable[[Array, Array], Array],
             params:dict,
             u:Callable[[Array, Array], Array],
             m:Callable[[PRNGKeyArray, Array, Array], Array], 
             s0:Array,
             T:int
             ) -> float:
    '''
    Simulates multiple paths given current policy function and
    save the results. This function is used after params have been
    optimized via train(). 

    Parameters:
    ----------
    key: JAX random number key
    policy: Policy function. Given a state, returns the action. 
    params: NN parameters
    u: Instantaneous utility function
    m: Law of motion
    s0: Initial state (N_simul x n_states)
    T: number of periods
    
    Returns:
    ----------
    array of simulation results, which comes from stacking the following
        intermediate arrays
            states: jnp.ndarray
                states in simulations (N_simul x N_states x T)
            actions: jnp.ndarray
                policy functions in simulations (N_simul x T)
            values: jnp.ndnarray
                value function in simulations (N_simul x T)
    '''
    N_simul, N_states = s0.shape
    states = jnp.zeros((N_simul, N_states, T))
    states = states.at[:,:,0].set(s0)
    actions = jnp.zeros((N_simul, T))
    values = jnp.zeros((N_simul, T))
    for t in range(T):
        key, subkey = jax.random.split(key)
        state = states[:,:,t]
        action = policy(state, params)
        actions = actions.at[:,[t]].set(action)
        values = values.at[:,:t+1].set(u(state, action) + values[:,:t+1])
        if t + 1 < T:
            states = states.at[:,:,t+1].set(m(subkey, state, action))
    outputs = [jnp.repeat(jnp.array(range(1, N_simul + 1)), T)]
    for k in range(N_states):
        outputs.append(states[:,k,:].reshape(-1,1))
    outputs.append(actions.reshape(-1,1))
    outputs.append(values.reshape(-1,1))
    return jnp.column_stack(outputs)
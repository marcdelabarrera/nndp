# Standard imports
from typing import Callable
from dataclasses import dataclass
from typing import Protocol
from functools import partial
import warnings

# Third party imports
import jax
from jax import Array
import jax.numpy as jnp
from jax.tree_util import Partial
import optax
import matplotlib.pyplot as plt
# Local imports
from .model import Model
from .policy import Policy


@partial(jax.jit, static_argnames=['model'])
def evaluate_policy(policy: Callable[[Array], Array],
                    model: Model, 
                    k: Array,
                    x0: Array,
                    ) -> Array:
    '''
    Calculates the value function for each of k=1,...,K paths of 
    exogeneous states, k,  with initial condition of endogeneous
    states, x0.
    
    
    Returns:
    ----------
    Vector of value functions at for each path k:
        \sum_{t=0}^T \beta^t u(action_t, state_t(k))
    '''
    if x0.ndim == 1:
        x0 = x0[jnp.newaxis,:]
    if k.ndim == 2:
        k = k[jnp.newaxis,:,:]
    V, state = jnp.zeros((k.shape[0])), jnp.column_stack([k[:,0], x0])
    return jax.lax.fori_loop(0, model.T+1, 
                             body_fun = Partial(time_iteration, k=k, policy=policy, model=model), 
                             init_val = (V,state))[0]

def time_iteration(t, 
                   x:tuple[Array, Array],
                   policy: Callable[[Array], Array],
                   model:Model,
                   k: Array):
    V, state = x
    action = policy(state)
    V += model.u(state, action)
    state_next = model.m(state, action, k[:,t+1,:])
    return V, state_next

def train_step(policy:Policy,
               model: Model,
               k:Array,
               x0:Array,
               optimizer,
               opt_state
               ):
    '''
    Performs one optimization step to update neural network parameters, params,
    where objective function is -1 times the equal-weighted mean of value
    functions across K different paths of exogeneous states and initial 
    conditions.
    
    Parameters:
    ----------
    params: dictionary of neural network parameters
    u: Instantaneous utility function
    v_T: terminal value function at t=T
    m: Law of motion
    k: K x T+1 x n_exstates array, where n_exstates is the number of 
        exogeneous states
    x0: K x n_edstates array, where n_edstates is the number of 
        endogeneous states
    T: number of periods - 1, where t=0,...,T and t=T is death
    optimizer: optax optimizer object
    opt_state: state of optax optimizer
    
    Returns:
    ----------
    params: updated neutral network parameters
    opt_state: updated state of optimizer
    value: objective function value at previous step
    '''
    value, grads = jax.value_and_grad(lambda params: 
            -jnp.mean(evaluate_policy(policy = Partial(policy.nn, params = params),
                                            model= model,
                                            k = k,
                                            x0 = x0
                                            )
                                )
    )(policy.params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(policy.params, updates)
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

def train_agnostic(policy: Policy,
                    model: Model,
                    k: Array,
                    x0: Array,
                    batch_size: int = 10,
                    optimizer = optax.adam(1e-4),
                    epochs: int = None,
                    )->tuple[Policy,TrainResult]:
    '''
    Optimizes parameters of a neural network, policy, by performing multiple steps
    of stochastic gradient descent. In each step, a batch of size batch_size is
    used from the path of exogeneous states passed, k. Training completes when
    all batches have been used n_cycles times. 
    
    Parameters:
    ----------
    params: dictionary of initialized neural network parameters
    policy: neural network to optimize
    u: Instantaneous utility function
    v_T: terminal value function at t=T
    m: Law of motion
    k: K x T+1 x n_exstates array, where n_exstates is the number of 
        exogeneous states
    x0: K x n_edstates array, where n_edstates is the number of 
        endogeneous states
    T: number of periods - 1, where t=0,...,T and t=T is death
    batch_size: number of rows of k to pass in each training step
    optimizer: optax optimizer object
    n_cycles: number of times to cycle through exogeneous paths, k
    
    Returns:
    ----------
    params: neural network parameters at final step of optimization
    '''
    
    if k.shape[0] != x0.shape[0]:
        raise ValueError('k and x0 must have the same number of paths')
    if k.shape[1] != model.T + 1:
        raise ValueError('k must have T+1 columns')
    
    epochs = epochs if epochs is not None else k.shape[0]//batch_size
    if epochs*batch_size > k.shape[0]:
        key = jax.random.PRNGKey(0)
        indices = jax.random.randint(key, (epochs*batch_size,), 0, k.shape[0])
        k = k[indices]
        x0 = x0[indices]

    opt_state = optimizer.init(policy.params)
    values = jnp.empty(epochs)


    for n in range(0, epochs*batch_size, batch_size):
        k_batch = k[n:n+batch_size]
        x0_batch = x0[n:n+batch_size]
        policy.params, opt_state, value = train_step(policy = policy, 
                                                model = model,
                                                k = k_batch,
                                                x0 = x0_batch,
                                                optimizer = optimizer, 
                                                opt_state = opt_state)
        values = values.at[n//batch_size].set(value)
        print(f'\rIteration {n//batch_size}, ({n/(epochs*batch_size)*100:.2f} %), value:{-value:.6f}', end = '')
    return policy, TrainResult(-values)


import jax
import jaxlib
import jax.numpy as jnp
from jax._src.basearray import Array
from jax.tree_util import Partial
import haiku as hk
from typing import Callable

def policy_nn(X:Array,
              P:int, 
              N_nodes:int,
              N_hidden:int,
              f_activation:Callable,
              f_outputs:list
              ) -> Array:
    '''
    This function defines a policy function that maps a matrix of states (exogeneous and endogeneous) into the 
    model policy functions. The policy function is a neural network. 

    Parameters:
    ----------
    X: N x K matrix of states, where K is the number of states and the function is vectorized over the N rows
    P: dimension of output layer = dimension of policy function
    N_nodes: number of nodes in each layer except output layer
    N_layers: number of hidden layers
    f_activation: activation function use in all but output layers (e.g. jax.nn.tanh, jax.nn.relu)
    f_outputs: list of Callable jaxlib.xla_extension.CompiledFunction (e.g. jax.nn.sigmoid). 
        The length of this list must be equal to P, as you need a separate activation function for each output. 
        
    Returns:
    ----------
    N x P matrix of policies, where P is dimension of policy function
    '''
    # Input layer + hidden laters
    for i in range(N_hidden + 1):
        X = hk.Linear(N_nodes)(X)
        X = f_activation(X)
    # Output layer
    X = hk.Linear(P)(X)
    Y = [f_outputs[p](X[:, p]) for p in range(P)]        
    return jnp.column_stack(Y)

def initialize_nn(key:Array,
                  K:int, 
                  P:int, 
                  N_nodes:int,
                  N_hidden:int,
                  f_activation:jaxlib.xla_extension.CompiledFunction,
                  f_outputs:list
                  ) -> list[dict, Callable]:
    '''
    This function initializes a Haiku Neural Network policy function with the given set of parameters. 
    
    Parameters:
    ----------
    key: JAX RNG key
    K: number of state variables in model
    P: dimension of output layer = dimension of policy function
    N_nodes: number of nodes in each layer except output layer
    N_layers: number of hidden layers
    f_activation: activation function use in all but output layers (e.g. jax.nn.tanh, jax.nn.relu)
    f_outputs: list of Callable jaxlib.xla_extension.CompiledFunction (e.g. jax.nn.sigmoid). 
        The length of this list must be equal to P, as you need a separate activation function for each output. 
        
    Returns:
    ----------
    params: initialized parameters of NN
    nn: policy function that is a neural network
    '''  
    # HK transform to get policy function that takes parameters as inputs
    init, policy_args = hk.without_apply_rng(hk.transform(policy_nn))
    # Initialize parameters
    x0 = jnp.column_stack([0.]*K)
    key, subkey = jax.random.split(key)
    params = init(subkey, x0, P, N_nodes, N_hidden, f_activation, f_outputs)
    # Make a function that surpress NN parameters
    def nn(p, X):
        return policy_args(p, X, P, N_nodes, N_hidden, f_activation, f_outputs)
    return params, nn

def make_policy_function(nn_to_action:Callable,
                         key:Array,
                         K:int, 
                         P:int, 
                         N_nodes:int,
                         N_hidden:int,
                         f_activation:jaxlib.xla_extension.CompiledFunction,
                         f_outputs:list
                         ) -> Callable:
    '''
    This function creates a policy function for the model of interest that is Haiku Neural
    Network with the supplied set of parameters.
    
    Parameters:
    ----------
    nn_to_action: user-defined function that characterizes how outputs of neural network
        translate into an action in the model
    key: JAX RNG key
    K: number of state variables in model
    P: dimension of output layer = dimension of policy function
    N_nodes: number of nodes in each layer except output layer
    N_layers: number of hidden layers
    f_activation: activation function use in all but output layers (e.g. jax.nn.tanh, jax.nn.relu)
    f_outputs: list of Callable jaxlib.xla_extension.CompiledFunction (e.g. jax.nn.sigmoid). 
        The length of this list must be equal to P, as you need a separate activation function for each output. 
        
    Returns:
    ----------
    params: initialized parameters of NN
    policy: policy function that is a neural network with signature policy(state, params)
    '''
    params, nn = initialize_nn(key = key,
                               K = K,
                               P = P,
                               N_nodes = N_nodes,
                               N_hidden = N_hidden,
                               f_activation = f_activation,
                               f_outputs = f_outputs
                               )
    policy = Partial(nn_to_action, nn = Partial(nn))
    return params, policy
# Standard imports
from typing import Callable
from dataclasses import dataclass
# Third party imports
import jax
import jaxlib
import jax.numpy as jnp
from jax._src.basearray import Array
from jax.tree_util import Partial
import haiku as hk


def deep_nn(x:Array,
            n_actions:int, 
            nodes_per_layer:int,
            hidden_layers:int,
            hidden_activation:Callable,
            output_activation:list[Callable]
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
    for layer in range(hidden_layers + 1):
        x = hk.Linear(nodes_per_layer)(x)
        x = hidden_activation(x)
    # Output layer
    x = hk.Linear(n_actions)(x)
    y = [output_activation[p](x[:, p]) for p in range(n_actions)]        
    return jnp.column_stack(y)



def initialize_deep_nn(key:Array,
                       n_states:int, 
                       n_actions:int, 
                        nodes_per_layer:int,
                        hidden_layers:int,
                        hidden_activation:jaxlib.xla_extension.PjitFunction,
                        output_activation:list
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
    f_outputs: list of Callable jaxlib.xla_extension.PjitFunction (e.g. jax.nn.sigmoid). 
        The length of this list must be equal to P, as you need a separate activation function for each output. 
        
    Returns:
    ----------
    params: initialized parameters of NN
    nn: policy function that is a neural network
    '''  
    # HK transform to get policy function that takes parameters as inputs
    init, policy_args = hk.without_apply_rng(hk.transform(deep_nn))
    # Initialize parameters
    x0 = jnp.zeros((1,n_states))
    key, subkey = jax.random.split(key)
    params = init(subkey, x0, n_actions, nodes_per_layer, hidden_layers, hidden_activation, output_activation)
    # Make a function that surpress NN parameters
    def nn(params, x):
        return policy_args(params, x, n_actions, nodes_per_layer, hidden_layers, hidden_activation, output_activation)
    return params, nn

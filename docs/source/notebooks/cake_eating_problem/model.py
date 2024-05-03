import jax
import jax.numpy as jnp

def u(state:jax.Array, action:jax.Array)->jax.Array:
    
  '''
  Reward function given a state and action
  '''
  s = state[...,1]
  c = action[...,0]
  return jnp.log(s*c).reshape(-1,1)

def m(key:jax.random.PRNGKey,state, action):
  '''
  Given a state and an action, returns a (possibly stockastic) state s_t+1
  '''
  t, s = state[...,0], state[...,1]
  c = action[...,0]
  return jnp.column_stack([t+1, s*(1-c)])

def policy(state, params, nn):
  return nn(params, state)

T = 5

def F(key:jax.random.PRNGKey, N:int):
  '''
  Samples initial distribution of states. t=0 and some cake size from 0 to 1.
  '''
  t = jnp.zeros(shape=(N,1))
  return jnp.column_stack([t,jax.random.uniform(key, shape = (N,1))])


from jax import Array


from .model import Model
from .samplers import Sampler
from .policy import Policy
from .train import evaluate_policy



def evaluate_state(state:Array, policy:Policy, model:Model, sampler:Sampler, n:int=10, seed=None)->Array:
    x0, k = sampler.path(initial_state = state, n=n, seed=seed)
    return evaluate_policy(policy, model, k, x0).mean()
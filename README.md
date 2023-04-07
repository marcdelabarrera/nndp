# Dynamic Programming with Neural Networks `(nndp)`

Marc de la Barrera i Bardalet, Tim de Silva

`nndp` provides a framework for solving finite horizon dynamic programming problems using neural networks that is implemented using the [JAX](https://github.com/google/jax) functional programming paradigm and [Haiku](https://github.com/deepmind/dm-haiku). This solution technique, introduced and described in detail by [Duarte, Fonesca, Goodman, and Parker (2021)](https://0f2486b1-f568-477b-8307-dd98a6c77afd.filesusr.com/ugd/f9db9d_972da014adb2453b8a4dab0239909062.pdf), applies to problems of the following form: 

$$V(s_0)=\max_{a_t\in\Gamma(s_t)} E_0\left[\sum_{t=0}^T u(s_t,a_t)\right],$$

$$s_{t+1}=m(s_{t},a_{t},\epsilon_t), $$

$$s_0 \sim F(\cdot).$$

The state vector is denoted by $s_t=(k_t, x_t)$, where $k_t$ are exogenous states and $x_t$ are endogenous states. We adopt the convention that the first exogenous state in $k_t$ is $t$. The goal is to find a policy function $\pi(s_t)$ that satisfies:

$$\hat V(s_0,\pi)=E_0\left[\sum_{t=0}^T u(s_t,\pi(s_t))\right],$$

$$s_{t+1}=m(s_{t},\pi(s_{t}),\epsilon_t),$$

$$V(s_0)=\hat V(s_0,\pi)\quad \forall s_0.$$

We parametrize $\pi(s_t)=\tilde\pi(s_t,\theta)$ as a fully connected feedforward neural network and update the networks’ parameters, $\theta$, using stochastic gradient descent. To use this framework, the user only needs to write the following functions that are defined by the dynamic programming problem of interest:

1. `u(state, action)`: reward function for $s_t$ = `state` and $a_t$ = `action`
2. `m(key, state, action)`: state evolution equation for $s_{t+1}$ if $s_t$ = `state` and $a_t$ = `action`. `key` is a JAX RNG key used to simulate any shocks present in the model.
3. `Gamma(state)`: defines the set of possible actions, $a_t$, at $s_t$ = `state`
4. `F(key, N)`: samples `N` observations from the distribution of $s_0$. `key` is a JAX RNG key used to simulate any shocks present in the model.
5. `nn_to_action(state, params, nn)`: defines how the output of a Haiku Neural Network, `nn`, with parameters, `params`, is mapped into an action at $s_t$ = `state`

We provide an example application to the income fluctations problem in `examples/income_fluctuations/main.ipynb` to illustrate how this framework can be used.

# References
Duarte, Victor, Julia Fonseca, Jonathan A. Parker, and Aaron Goodman (2021), Simple Allocation Rules and Optimal Portfolio Choice Over the Lifecycle, Working Paper.


# Dynamic Programming with Neural Networks `(nndp)`

Marc de la Barrera i Bardalet, Tim de Silva

`nndp` provides a framework for solving finite horizon dynamic programming problems using neural networks, implemented using JAX and Haiku. This solution technique, introduced and described in detail by Duarte, Fonesca, Goodman, and Parker (2021), applies to problems of the following form: 

$$V(s_0)=\max_{a_t\in\Gamma(s_t)} E_0\left[\sum_{t=0}^T u(s_t,a_t)\right],$$

$$s_{t+1}=m(s_{t},a_{t},\epsilon_t), $$

$$s_0 \sim F(\cdot).$$

The state vector is denoted by $s_t=(k_t, x_t)$, where $k_t$ are exogenous states and $x_t$ are endogenous states. We adopt the convention that the first exogenous state in $k_t$ is $t$. The goal is to find a policy function $\pi(s_t)$ that defines the optimal action $a_t=\pi(s_t)$. We parametrize $\pi(s_t)=\tilde\pi(s_t,\theta)$ as a fully connected feedforward neural network and update the networks’ parameters, $\theta$, using stochastic gradient descent.

To use this framework,

We provide an example application to the income fluctations problem in `examples/`. 


$$\hat V(s_0,\pi)=E_0\left[\sum_{t=0}^T u(s_t,\pi(s_t))\right]$$

$$s_{t+1}=m(s_{t},\pi(s_{t}),\epsilon_t)$$

The goal is to find $\pi$ such that:

$$V(s_0)=\hat V(s_0,\pi)\qquad \forall s_0$$

Define `m(state,action)` that returns next state, `Gamma(state)` that returns intervals for the action and `u(state,action)` that returns the time 0 utility.


# References
Duarte, Victor, Julia Fonseca, Jonathan A. Parker, and Aaron Goodman (2021), “Simple Allocation Rules and Optimal Portfolio Choice Over the Lifecycle”, Working Paper.

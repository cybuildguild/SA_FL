from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jrd

import optax
import optax.tree

# Variables involved:
    # iteration, k
    # parameters w
    # gradients g
    # stepsize n
    # preconditioning matrix P
        # Approx. of Hessian inverse

dimension = 8

def funct(w):
    return jnp.sum(jnp.abs(w))

# Linesearch
linesearch = optax.scale_by_backtracking_linesearch(max_backtracking_steps=15, store_grad=True)
# # Can also do below
# linesearch = optax.scale_by_zoom_linesearch(max_linesearch_steps=15, store_grad=True)

# Optimizer
opt = optax.chain(
    optax.sgd(learning_rate=1.0),
    # Compare with or without linesearch by commenting this line
    linesearch,
)

# Initialize optimization
w = jrd.normal(jrd.PRNGKey(0), (dimension,))
state = opt.init(w)

# Run optimization 
for i in range(16):
    # Computes the function value and its gradient
        # Returns a tuple
    v, g = jax.value_and_grad(funct)(w)
    print(f'Iteration: {i}, Value: {v:.2e}')
    # Transforms gradient into parameter update
        # Updates optimizer's state
    u, state = opt.update(g, state, w, value=v, grad=g, value_fn=funct)
    w = w + u

print(f'Final value: {funct(w):.2e}')
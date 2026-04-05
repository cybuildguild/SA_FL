# L-BFGS: Limited-memory Broyden-Fletcher-Goldfarb-Shanno Algorithm
    # An optimization method

"""
Resources:
1) https://optax.readthedocs.io/en/stable/_collections/examples/lbfgs.html
2) https://en.wikipedia.org/wiki/Accelerated_Linear_Algebra
3) https://medium.com/@khang.pham.exxact/intro-to-jax-for-machine-learning-9e6b11738f02
"""

from typing import NamedTuple

# Import jax
"""
Uses XLA (Accelerated Linear Algebra) to compile NumPy functions [3]
XLA optimizes computation graphs for execution [2]
"""
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

# Using L-BFGS as a Gradient Transformation

vector_dimension = 8
optimizer = jnp.ones(vector_dimension)
matrix = jrd.normal(jrd.PRNGKey(0), (vector_dimension, vector_dimension))
matrix = matrix.dot(matrix.T)

def funct(w):
    return 0.5 * (w - optimizer).dot(matrix.dot(w - optimizer))

# Define optimizer 
lr = 1e-1 
opt = optax.scale_by_lbfgs()

# Initialize optimization
w = jrd.normal(jrd.PRNGKey(1), (vector_dimension,))
state = opt.init(optimizer)

# Run optimization 
for i in range(16):
    v, g = jax.value_and_grad(funct)(w)
    print(f'Iteration: {i}, Value: {v:.2e}')
    u, state = opt.update(g, state, w)
    w = w - lr * u

print(f'Final value: {funct(w):.2e}')
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

dimension = 8
# Create an array with dimension size of variable "dimension"
    # As is, this array will be a 1D array full of #"dimension" ones
true_min = jnp.ones(dimension)
# Generate a sample of random normal numbers
    # Needs a key to be "pure" (same input --> same output)
        # Same key --> Same array generated
# Produces a matrix of size dimension x dimension
    # where each element is a randomly sampled number
matrix = jrd.normal(jrd.PRNGKey(0), (dimension, dimension))
# Compute the dot product of matrix and the transpose of matrix
    # Creates a symmetric matrix --> produces REAL eigenvalues
matrix = matrix.dot(matrix.T)

def funct(w):
    # w - true_min ==> satisfies the SECANT CONDITION 
        # As true_min is the parameter before the current parameter (w)
    return 0.5 * (w - true_min).dot(matrix.dot(w - true_min))

# Define optimizer 

# Where lr is the stepsize (or learning rate)
lr = 1e-1 
opt = optax.scale_by_lbfgs()

# Initialize optimization
w = jrd.normal(jrd.PRNGKey(1), (dimension,))
state = opt.init(true_min)

# Run optimization 
for i in range(16):
    v, g = jax.value_and_grad(funct)(w)
    print(f'Iteration: {i}, Value: {v:.2e}')
    u, state = opt.update(g, state, w)
    w = w - lr * u

print(f'Final value: {funct(w):.2e}')
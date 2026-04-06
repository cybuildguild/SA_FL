# Combines gradient transformation of L-BFGS and linesearch

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jrd

import optax
import optax.tree

def run_opt(init_params, funct, opt, max_iter, tol):
    value_and_grad_fct = optax.value_and_grad_from_state(funct)

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fct(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=funct
        )
        params = optax.apply_updates(params, updates)
        return params, state
    
    def continuing_criterion(carry):
        _, state = carry
        iter_num = optax.tree.get(state, 'count')
        grad = optax.tree.get(state, 'grad')
        err = optax.tree.norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))
    
    init_carry = (init_params, opt.init(init_params))
    final_params, final_state = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )

    return final_params, final_state
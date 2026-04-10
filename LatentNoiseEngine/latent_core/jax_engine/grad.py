"""
latent_core/jax_engine/grad.py
================================
Gradient-based optimisation utilities.

Used for:
  - Tuning QSP phase solver
  - Optimising polynomial coefficients
  - Training latent-to-channel mapping weights
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
import optax
from typing import Callable


def make_optimizer(name: str = "adam", lr: float = 1e-3) -> optax.GradientTransformation:
    """Factory for common optimisers."""
    optimisers = {
        "adam": optax.adam(lr),
        "sgd": optax.sgd(lr),
        "rmsprop": optax.rmsprop(lr),
        "adamw": optax.adamw(lr),
    }
    if name not in optimisers:
        raise ValueError(f"Unknown optimiser: {name}")
    return optimisers[name]


def gradient_step(
    loss_fn: Callable,
    params,
    opt_state,
    optimizer: optax.GradientTransformation,
) -> tuple:
    """
    Single gradient descent step.

    Returns
    -------
    (loss, params_next, opt_state_next)
    """
    loss_val, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state_next = optimizer.update(grads, opt_state)
    params_next = optax.apply_updates(params, updates)
    return float(loss_val), params_next, opt_state_next


def optimize(
    loss_fn: Callable,
    init_params,
    n_steps: int = 500,
    lr: float = 1e-3,
    optimizer_name: str = "adam",
    tol: float = 1e-8,
    verbose: bool = False,
) -> tuple:
    """
    Run gradient-based optimisation until convergence or max steps.

    Returns
    -------
    (best_params, final_loss, loss_history)
    """
    optimizer = make_optimizer(optimizer_name, lr)
    params = init_params
    opt_state = optimizer.init(params)
    loss_jit = jax.jit(jax.value_and_grad(loss_fn))

    history = []
    best_params = params
    best_loss = float("inf")

    for step in range(n_steps):
        loss_val, grads = loss_jit(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        loss_f = float(loss_val)
        history.append(loss_f)

        if loss_f < best_loss:
            best_loss = loss_f
            best_params = params

        if verbose and step % 100 == 0:
            print(f"  step {step:4d} | loss = {loss_f:.6e}")

        if loss_f < tol:
            break

    return best_params, best_loss, history

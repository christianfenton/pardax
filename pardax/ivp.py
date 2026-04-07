import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .timesteppers import StepperLike


def solve_ivp(
    fun: Any,
    t_span: tuple[float, float],
    y0: Float[Array, "*state"],
    stepper: StepperLike,
    step_size: float,
    args: tuple = (),
    num_checkpoints: int = 0,
) -> tuple[Float[Array, " steps"], Float[Array, " steps *state"]]:
    """
    Integrate dy/dt = fun(t, y, *args) from t_span[0] to t_span[1].

    Uses `jax.lax.scan` internally and is compatible with all JAX
    transformations, including reverse-mode differentiation (`jax.grad`).

    Output times are equally spaced: the interval is divided into
    `num_checkpoints + 1` segments and a snapshot is saved at the end of
    each segment, plus the initial state.

    Args:
        fun: Right-hand side dy/dt = fun(t, y, *args)
        t_span: Tuple of start and end time, e.g. (t_start, t_end)
        y0: Initial condition at t_span[0]
        stepper: Time-stepper instance (e.g., RK4(), BackwardEuler())
        step_size: Maximum time step size
            The actual step may be smaller to fit an integer number of steps
        args: Additional arguments passed to fun
        num_checkpoints: Number of equally spaced intermediate snapshots
            to store between t_start and t_end. If 0 (default),
            only the initial and final states are returned.

    Returns:
        t: Time points, shape ``(num_checkpoints + 2,)``
        y: Solution snapshots, shape ``(num_checkpoints + 2, *y0.shape)``
    """
    t_start, t_end = t_span

    # Number of steps in inner and outer loops
    num_steps_total = math.ceil((t_end - t_start) / step_size)
    num_steps_outer = num_checkpoints + 1
    num_steps_inner = math.ceil(num_steps_total / num_steps_outer)

    # Actual time step size
    h = jnp.asarray((t_end - t_start) / num_steps_total)

    def inner(carry, _):
        t, y, step = carry

        t_new, y_new = jax.lax.cond(
            step < num_steps_total,
            lambda t, y: (t + h, stepper.step(fun, t, y, h, args)),
            lambda t, y: (t, y),
            t,
            y,
        )

        return (t_new, y_new, step + 1), None

    def outer(carry, _):
        t, y, step = carry

        (t, y, step), _ = jax.lax.scan(
            inner, (t, y, step), length=num_steps_inner
        )

        return (t, y, step), (t, y)

    _, (t_all, y_all) = jax.lax.scan(
        outer, (t_start, y0, 0), length=num_steps_outer
    )

    # Prepend initial state
    t_all = jnp.concatenate([jnp.array([t_start]), t_all])
    y_all = jnp.concatenate([y0[None], y_all], axis=0)

    return t_all, y_all


def integrate(
    fun: Any,
    t_eval: Float[Array, " steps"],
    y0: Float[Array, "*state"],
    stepper: StepperLike,
    step_size_fn: Callable[..., float],
    args: tuple = (),
) -> tuple[Float[Array, " steps"], Float[Array, " steps *state"]]:
    """
    Integrate dy/dt = fun(t, y, *args), returning states at times t_eval.

    The time step size at each sub-step is given by
    ``step_size_fn(t, y, *args) -> h``, which may depend on the current
    time or solution state (e.g. a CFL condition). Steps are clipped to
    avoid overshooting each target time.

    Uses `jax.lax.while_loop` internally. Compatible with `jax.jit`,
    `jax.vmap`, and forward-mode differentiation (`jax.jvp`), but
    **not** reverse-mode differentiation (`jax.grad`). Use
    [solve_ivp][pardax.solve_ivp] when reverse-mode autodiff is required.

    Args:
        fun: Right-hand side dy/dt = fun(t, y, *args)
        t_eval: Sorted array of output times
        y0: Initial condition at t_eval[0]
        stepper: Time-stepper instance (e.g., RK4(), BackwardEuler())
        step_size_fn: Callable ``(t, y, *args) -> h`` returning the
            desired step size from the current state
        args: Additional arguments passed to fun and step_size_fn

    Returns:
        t: Time points, shape ``(len(t_eval),)``
        y: Solution snapshots, shape ``(len(t_eval), *y0.shape)``
    """
    eps = jnp.finfo(t_eval.dtype).eps

    def step_to_time(carry, scan_input):
        t, y = carry
        t_target = scan_input

        def cond_fn(state):
            t, _ = state
            return t < t_target - eps

        def body_fn(state):
            t, y = state
            h = step_size_fn(t, y, *args)
            h = jnp.minimum(h, t_target - t)
            h = jnp.maximum(h, 0)
            y_new = stepper.step(fun, t, y, h, args)
            return (t + h, y_new)

        t, y = jax.lax.while_loop(cond_fn, body_fn, (t, y))

        return (t, y), (t, y)

    _, (t_all, y_all) = jax.lax.scan(
        step_to_time, (t_eval[0], y0), (t_eval[1:])
    )

    # Prepend initial state
    t_all = jnp.concatenate([t_eval[0:1], t_all])
    y_all = jnp.concatenate([y0[None], y_all], axis=0)

    return t_all, y_all

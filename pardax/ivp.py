import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .timesteppers import StepperLike


def solve_ivp(
    fun: Callable,
    t_span: tuple[float, float],
    y0: Float[Array, "..."],
    stepper: StepperLike,
    step_size: float,
    params: Any = None,
    num_checkpoints: int = 0,
) -> tuple[Float[Array, " steps"], Float[Array, "steps ..."]]:
    """
    Integrate dy/dt = fun(t, y, params) from t_span[0] to t_span[1].

    Uses `jax.lax.scan` internally and is compatible with all JAX
    transformations, including reverse-mode differentiation (`jax.grad`).

    Output times are equally spaced: the interval is divided into
    `num_checkpoints + 1` segments and a snapshot is saved at the end of
    each segment, plus the initial state.

    Args:
        fun: Right-hand side dy/dt = fun(t, y, params)
        t_span: Tuple of start and end time, e.g. (t_start, t_end)
        y0: Initial condition at t_span[0]
        stepper: Time-stepper instance (e.g., RK4(), BackwardEuler())
        step_size: Maximum time step size.
            The actual step may be smaller to fit an integer number of steps
        params: Parameters pytree passed to fun
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

    step_size_actual = jnp.asarray((t_end - t_start) / num_steps_total)

    def inner(carry, _):
        t, y, num_steps, stepper = carry

        def do_step(args):
            t, y, stepper = args
            y, stepper = stepper(fun, t, y, step_size_actual, params)
            return t + step_size_actual, y, stepper

        def skip(args):
            return args

        t, y, stepper = jax.lax.cond(
            num_steps < num_steps_total, do_step, skip, (t, y, stepper)
        )

        return (t, y, num_steps + 1, stepper), None

    def outer(carry, _):
        t, y, num_steps, stepper = carry

        (t, y, num_steps, stepper), _ = jax.lax.scan(
            inner, (t, y, num_steps, stepper), length=num_steps_inner
        )

        return (t, y, num_steps, stepper), (t, y)

    _, (t_all, y_all) = jax.lax.scan(
        outer, (t_start, y0, 0, stepper), length=num_steps_outer
    )

    # Prepend initial state
    t_all = jnp.concatenate([jnp.array([t_start]), t_all])
    y_all = jnp.concatenate([y0[None], y_all], axis=0)

    return t_all, y_all


def integrate(
    fun: Callable,
    t_eval: Float[Array, " steps"],
    y0: Float[Array, "..."],
    stepper: StepperLike,
    step_size_fn: Callable[..., float],
    params: Any = None,
) -> tuple[Float[Array, " steps"], Float[Array, "steps ..."]]:
    """
    Integrate dy/dt = fun(t, y, params), returning states at times t_eval.

    The time step size at each sub-step is given by
    ``step_size_fn(t, y, params) -> step_size``,
    which may depend on the current time or solution (e.g. a CFL condition).
    Steps are clipped to avoid overshooting each target time.

    Uses `jax.lax.while_loop` internally, which is not supported by
    reverse-mode  automatic differentiation with `jax.grad`.
    Use [solve_ivp][pardax.solve_ivp] when reverse-mode autodiff is required.

    Args:
        fun: Right-hand side dy/dt = fun(t, y, params)
        t_eval: Sorted array of output times
        y0: Initial condition at t_eval[0] (any JAX pytree)
        stepper: Time-stepper instance (e.g., RK4(), BackwardEuler())
        step_size_fn: Callable ``(t, y, params) -> dt`` returning the
            desired step size from the current state
        params: Parameters pytree passed to fun and step_size_fn

    Returns:
        t: Time points, shape ``(len(t_eval),)``
        y: Solution snapshots, shape ``(len(t_eval), *y0.shape)``
    """
    eps = jnp.finfo(t_eval.dtype).eps

    def step_to_time(carry, scan_input):
        t, y, stepper = carry
        t_target = scan_input

        def cond_fn(s):
            t, _, _ = s
            return t < t_target - eps

        def body_fn(s):
            t, y, stepper = s
            step_size = step_size_fn(t, y, params)
            step_size = jnp.minimum(step_size, t_target - t)
            step_size = jnp.maximum(step_size, 0)
            y, stepper = stepper(fun, t, y, step_size, params)
            return (t + step_size, y, stepper)

        t, y, stepper = jax.lax.while_loop(cond_fn, body_fn, (t, y, stepper))

        return (t, y, stepper), (t, y)

    _, (t_all, y_all) = jax.lax.scan(
        step_to_time, (t_eval[0], y0, stepper), (t_eval[1:])
    )

    # Prepend initial state
    t_all = jnp.concatenate([t_eval[0:1], t_all])
    y_all = jnp.concatenate([y0[None], y_all], axis=0)

    return t_all, y_all

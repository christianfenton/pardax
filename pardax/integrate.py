"""Time integration using scan + fori_loop."""

from typing import Tuple, Union

import jax
from jax import Array
import jax.numpy as jnp

from .type_aliases import RHS, SplitRHS
from .stepper import AbstractStepper


def solve_ivp(
    fun: Union[RHS, SplitRHS],
    t_eval: Array,
    y0: Array,
    stepper: AbstractStepper,
    dt_max: float,
    args: tuple = ()
) -> Tuple[Array, Array]:
    """
    Integrate dy/dt = fun(t, y, *args), returning states at times t_eval.

    Args:
        fun: Right-hand side of system dy/dt = fun(t, y, *args)
        t_eval: Times at which to return the solution. Must be sorted
            and include the initial time as the first element.
        y0: Initial condition at t_eval[0]
        stepper: Time-stepper instance (e.g., RK4(), BackwardEuler())
        dt_max: Maximum time step size
        args: Additional arguments to pass to fun

    Returns:
        t: Array of time points, shape (len(t_eval),)
        y: Array of solution values, shape (len(t_eval), *y0.shape)

    Example usage:
    ```python
    import jax.numpy as jnp
    import pardax as pdx

    # Define ODE: dy/dt = -k*y
    def fun(t, y, k):
        return -k * y

    y0 = jnp.array([1.0])
    t_eval = jnp.linspace(0.0, 2.0, 5)
    k = 0.5

    t, y = pdx.solve_ivp(fun, t_eval, y0, pdx.RK4(), step_size=0.01, args=(k,))
    ```
    """
    t_eval = jnp.asarray(t_eval)

    gaps = jnp.diff(t_eval)
    steps_per_interval = jnp.ceil(gaps / dt_max).astype(int)
    dt_per_interval = gaps / steps_per_interval

    def step_to_time(carry, interval_info):
        t, y = carry
        n_steps, dt = interval_info

        def body_fn(_, state):
            t, y = state
            y_next = stepper.step(fun, t, y, dt, args)
            return (t + dt, y_next)

        t, y = jax.lax.fori_loop(0, n_steps, body_fn, (t, y))
        return (t, y), (t, y)

    _, (t_all, y_all) = jax.lax.scan(
        step_to_time, (t_eval[0], y0), (steps_per_interval, dt_per_interval)
    )

    # Prepend initial state
    t_all = jnp.concatenate([t_eval[0:1], t_all])
    y_all = jnp.concatenate([y0[None], y_all], axis=0)

    return t_all, y_all
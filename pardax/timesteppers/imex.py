from typing import Dict, Callable

import equinox as eqx
from jax import Array

from .base import AbstractStepper


class IMEX(eqx.Module):
    """
    Split implicit-explicit (IMEX) time-stepping scheme.

    Splits the ODE into stiff (implicit) and non-stiff (explicit) parts:
        dy/dt = f_explicit(t, y) + f_implicit(t, y)

    The scheme advances the solution in two steps:
        1. Explicit: y_star = y_curr + h * f_explicit(t_curr, u_curr)
        2. Implicit: y_next = y_star + h * f_implicit(t_next, y_next)

    Example:
        ```python
        import pardax

        def explicit_term(t, u, ...):
            return ...

        def implicit_term(t, u, ...):
            return ...

        # Instantiate stepper
        stepper = pardax.IMEX(
            implicit=pardax.BackwardEuler(), explicit=pardax.RK4()
        )

        # Define ODE as a dict
        ode = {'implicit': implicit_term, 'explicit': explicit_term}

        # Solve
        t, y = solve_ivp(ode, t_span, y0, stepper, step_size, args)
        ```
    """

    explicit: AbstractStepper
    implicit: AbstractStepper

    def step(
        self,
        fun: Dict[str, Callable[..., Array]],
        t: Array,
        y: Array,
        h: Array,
        args: tuple = (),
    ) -> Array:
        """Advance the solution by one time step.

        Args:
            fun: A dict with keys 'implicit' and 'explicit'.
            t: Current time.
            y: Current solution.
            h: Time step size.
            args: Additional arguments to pass to fun.

        Returns:
            Solution at t + h.
        """
        y_star = self.explicit.step(fun["explicit"], t, y, h, args)
        return self.implicit.step(fun["implicit"], t, y_star, h, args)

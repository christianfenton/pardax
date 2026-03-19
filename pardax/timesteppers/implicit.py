from typing import Callable

from jax import Array

from ..rootfinders import AbstractRootFinder, NewtonRaphson
from .base import AbstractStepper


class BackwardEuler(AbstractStepper):
    """Backward Euler time stepper."""

    root_finder: AbstractRootFinder

    def __init__(
        self, root_finder: AbstractRootFinder = NewtonRaphson()
    ) -> None:
        self.root_finder = root_finder

    def step(
        self,
        fun: Callable[..., Array],
        t: Array,
        y: Array,
        h: Array,
        args: tuple = (),
    ) -> Array:
        """Perform a single backward Euler step.

        Solves y_next = y_n + h * f(t_next, y_next, *args) at each time step.

        Args:
            fun: Right-hand side of system dy/dt = f(t, y, *args).
            t: Current time.
            y: Current solution.
            h: Time step size.
            args: Additional arguments to pass to fun.

        Returns:
            Solution at t + h.
        """

        def residual(y_next: Array) -> Array:
            return y_next - y - h * fun(t + h, y_next, *args)

        y0 = y + h * fun(t, y, *args)
        return self.root_finder(
            residual, y0, fun=fun, t=t + h, h=h, args=args, theta=1.0
        )

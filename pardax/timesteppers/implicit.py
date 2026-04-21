from collections.abc import Callable
from typing import Any

from jaxtyping import Array, Float

from ..rootfinders import AbstractRootFinder, NewtonRaphson
from .base import AbstractStepper


class BackwardEuler(AbstractStepper):
    """Backward Euler time stepper."""

    root_finder: AbstractRootFinder

    def __init__(
        self, root_finder: AbstractRootFinder = NewtonRaphson()
    ) -> None:
        self.root_finder = root_finder

    def __call__(
        self,
        fun: Callable,
        t: Float[Array, ""],
        y: Float[Array, "..."],
        step_size: Float[Array, ""],
        params: Any = None,
    ) -> tuple[Float[Array, "..."], "BackwardEuler"]:
        """Perform a single Backward Euler step.

        Solves y_next = y + step_size * fun(t + step_size, y_next, params).

        Args:
            fun: Right-hand side fun(t, y, params) -> dy/dt
            t: Current time
            y: Current solution
            step_size: Time step size
            params: Parameters passed through to fun

        Returns:
            Tuple of the new solution and stepper instance (y, stepper)
        """

        def residual(y_next):
            return y_next - y - step_size * fun(t + step_size, y_next, params)

        y_guess = y + step_size * fun(t, y, params)  # Forward Euler step

        y_new = self.root_finder(
            residual,
            y_guess,
            fun=fun,
            t=t + step_size,
            step_size=step_size,
            args=(params,),
            theta=1.0,
        )

        return y_new, self

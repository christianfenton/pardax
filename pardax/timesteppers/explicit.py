from collections.abc import Callable
from typing import Any

from jaxtyping import Array, Float

from .base import AbstractStepper


class ForwardEuler(AbstractStepper):
    """Forward Euler method."""

    def __call__(
        self,
        fun: Callable,
        t: Float[Array, ""],
        y: Float[Array, "..."],
        step_size: Float[Array, ""],
        params: Any = None,
    ) -> tuple[Float[Array, "..."], "ForwardEuler"]:
        """Perform a single Forward Euler step.

        Computes y_next = y + step_size * fun(t, y, params).

        Args:
            fun: Right-hand side fun(t, y, params) -> dy/dt
            t: Current time
            y: Current solution
            step_size: Time step size
            params: Parameters passed through to fun

        Returns:
            Tuple of the new solution and stepper instance (y, stepper)
        """
        return y + step_size * fun(t, y, params), self


class RK4(AbstractStepper):
    """Fourth (4th) order Runge-Kutta method."""

    def __call__(
        self,
        fun: Callable,
        t: Float[Array, ""],
        y: Float[Array, "..."],
        step_size: Float[Array, ""],
        params: Any = None,
    ) -> tuple[Float[Array, "..."], "RK4"]:
        """Perform a single RK4 step.

        Args:
            fun: Right-hand side fun(t, y, params) -> dy/dt
            t: Current time
            y: Current solution
            step_size: Time step size
            params: Parameters passed through to fun

        Returns:
            Tuple of the new solution and stepper instance (y, stepper)
        """
        k1 = fun(t, y, params)
        k2 = fun(t + 0.5 * step_size, y + 0.5 * step_size * k1, params)
        k3 = fun(t + 0.5 * step_size, y + 0.5 * step_size * k2, params)
        k4 = fun(t + step_size, y + step_size * k3, params)
        return y + (step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), self

from collections.abc import Callable

from jaxtyping import Array, Float

from .base import AbstractStepper


class ForwardEuler(AbstractStepper):
    """Forward Euler method."""

    def step(
        self,
        fun: Callable[..., Float[Array, "*state"]],
        t: Float[Array, ""],
        y: Float[Array, "*state"],
        h: Float[Array, ""],
        args: tuple = (),
    ) -> Float[Array, "*state"]:
        """Perform a single Forward Euler step.

        Computes y_next = y_curr + h * f(t_curr, y_curr, *args).

        Args:
            fun: Right-hand side of system dydt = f(t, y, *args)
            t: Current time
            y: Current solution
            h: Time step size
            args: Additional arguments to pass to fun

        Returns:
            Solution at t + h
        """
        return y + h * fun(t, y, *args)


class RK4(AbstractStepper):
    """Fourth (4th) order Runge-Kutta method."""

    def step(
        self,
        fun: Callable[..., Float[Array, "*state"]],
        t: Float[Array, ""],
        y: Float[Array, "*state"],
        h: Float[Array, ""],
        args: tuple = (),
    ) -> Float[Array, "*state"]:
        """Perform a single RK4 step.

        Args:
            fun: Right-hand side of system dy/dt = f(t, y, *args)
            t: Current time
            y: Current solution
            h: Time step size
            args: Additional arguments to pass to fun

        Returns:
            Solution at t + h
        """
        k1 = fun(t, y, *args)
        k2 = fun(t + 0.5 * h, y + 0.5 * h * k1, *args)
        k3 = fun(t + 0.5 * h, y + 0.5 * h * k2, *args)
        k4 = fun(t + h, y + h * k3, *args)
        return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

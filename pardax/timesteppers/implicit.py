from collections.abc import Callable

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

    def step(
        self,
        fun: Callable[..., Float[Array, "*state"]],
        t: Float[Array, ""],
        y: Float[Array, "*state"],
        h: Float[Array, ""],
        args: tuple = (),
    ) -> Float[Array, "*state"]:
        """Perform a single backward Euler step.

        Solves y_next = y_n + h * f(t_next, y_next, *args) at each time step.

        Args:
            fun: Right-hand side of system dy/dt = f(t, y, *args)
            t: Current time
            y: Current solution
            h: Time step size
            args: Additional arguments to pass to fun

        Returns:
            Solution at t + h
        """

        def residual(y_next: Float[Array, "*state"]) -> Float[Array, "*state"]:
            return y_next - y - h * fun(t + h, y_next, *args)

        y0 = y + h * fun(t, y, *args)
        return self.root_finder(
            residual, y0, fun=fun, t=t + h, h=h, args=args, theta=1.0
        )

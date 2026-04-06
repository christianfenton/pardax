"""Abstract base class for root-finding algorithms."""

import abc
from collections.abc import Callable

import equinox as eqx
from jaxtyping import Array, Float


class AbstractRootFinder(eqx.Module):
    """Base class for root-finding algorithms."""

    @abc.abstractmethod
    def __call__(
        self,
        residual_fn: Callable[
            [Float[Array, "*state"]], Float[Array, "*state"]
        ],
        y_guess: Float[Array, "*state"],
        fun: Callable[..., Float[Array, "*state"]],
        t: Float[Array, ""],
        h: Float[Array, ""],
        args: tuple,
        theta: float = 1.0,
    ) -> Float[Array, "*state"]:
        """
        Find the root of residual_fn(y) = 0.

        Args:
            residual_fn: Function mapping y -> R(y), where we seek R(y) = 0
            y_guess: Initial guess for the solution
            fun: Right-hand side function dy/dt = fun(t, y, *args)
            t: Time at which to evaluate the linearisation
            h: Time step size
            args: Additional arguments to pass to fun
            theta: Implicit coefficient. The system matrix is built as
                (I - theta * h * L). For backward Euler theta=1.0,
                for Crank-Nicolson theta=0.5, etc. Passed by the
                time stepper; users do not set this directly.

        Returns:
            Solution y such that residual_fn(y) ≈ 0
        """
        raise NotImplementedError

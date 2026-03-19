"""Abstract base class for root-finding algorithms."""

import abc
from typing import Callable

import equinox as eqx
from jax import Array


class AbstractRootFinder(eqx.Module):
    """Base class for root-finding algorithms."""

    @abc.abstractmethod
    def __call__(
        self,
        residual_fn: Callable[[Array], Array],
        y_guess: Array,
        fun: Callable[..., Array],
        t: Array,
        h: Array,
        args: tuple,
        theta: float = 1.0,
    ) -> Array:
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

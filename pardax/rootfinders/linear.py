"""Linear root finder and operator types for linear implicit systems."""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..linalg import AbstractLinearOperator, AbstractLinearSolver

from .base import AbstractRootFinder


class LinearRootFinder(AbstractRootFinder):
    """
    Single-step root finder for linear systems.

    Attributes:
        linsolver: Linear solver to use
        operator: Constructs the linear system of equations
    """

    linsolver: AbstractLinearSolver
    operator: AbstractLinearOperator

    def __init__(
        self,
        linsolver: AbstractLinearSolver,
        operator: AbstractLinearOperator,
    ) -> None:
        self.linsolver = linsolver
        self.operator = operator

    def __call__(
        self,
        residual_fn: Callable[
            [Float[Array, "*state"]], Float[Array, "*state"]
        ],
        y_guess: Float[Array, "*state"],
        fun: Any,
        t: Float[Array, ""],
        h: Float[Array, ""],
        args: tuple,
        theta: float = 1.0,
    ) -> Float[Array, "*state"]:
        """
        Solve for R(y) = 0 in a single step.

        Args:
            residual_fn: Residual function R(y) = y - y_n - theta * h * L(y)
            y_guess: Initial guess (used for shape / initial value)
            fun: Right-hand side of system (unused)
            t: Current time
            h: Time step size
            args: Additional arguments
            theta: Implicit coefficient (see AbstractRootFinder)

        Returns:
            Solution y such that residual_fn(y) ≈ 0
        """
        b = -residual_fn(jnp.zeros_like(y_guess))
        A = self.operator.system(t, theta * h, args)
        return self.linsolver(A, b, x0=y_guess)

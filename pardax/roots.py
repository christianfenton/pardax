"""Root-finding algorithms for implicit time-stepping schemes."""

import abc
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .linearise import AbstractLineariser, AutoJVP


class AbstractRootFinder(eqx.Module):
    """Base class for root-finding algorithms."""

    @abc.abstractmethod
    def __call__(
        self,
        residual_fn: Callable[[Array], Array],
        y_guess: Array,
        fun: Callable,
        t: Array,
        h: Array,
        args: tuple,
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

        Returns:
            Solution y such that residual_fn(y) ≈ 0
        """
        raise NotImplementedError


class NewtonRaphson(AbstractRootFinder):
    """
    Newton-Raphson root-finding algorithm.

    Iterative update: y_new = y_curr - J^{-1}(y_curr) * R(y_curr)

    The lineariser handles both the Jacobian construction and
    the linear solve, so NewtonRaphson only contains the outer
    iteration logic.

    Attributes:
        lineariser: Strategy for linearising and solving the Newton system
        tol: Convergence tolerance for residual norm
        maxiter: Maximum number of Newton-Raphson iterations
    """
    lineariser: AbstractLineariser
    tol: float = eqx.field(static=True)
    maxiter: int = eqx.field(static=True)

    def __init__(
        self,
        lineariser: AbstractLineariser = AutoJVP(),
        tol: float = 1e-6,
        maxiter: int = 50,
    ):
        self.lineariser = lineariser
        self.tol = tol
        self.maxiter = maxiter

    def __call__(
        self,
        residual_fn: Callable[[Array], Array],
        y_guess: Array,
        fun: Callable,
        t: Array,
        h: Array,
        args: tuple,
    ) -> Array:
        """
        Find the root of residual_fn(y) = 0 using Newton-Raphson method.

        Args:
            residual_fn: Residual function R(y)
            y_guess: Initial guess
            fun: Right-hand side function (passed to lineariser)
            t: Time at which to evaluate the linearisation
            h: Time step size
            args: Additional arguments to pass to fun

        Returns:
            Solution y
        """
        solve_linear = self.lineariser(fun, t, h, args)

        y_k = y_guess
        r_k = residual_fn(y_k)
        state0 = (y_k, r_k, 0)

        def body_fun(state):
            y_k, r_k, k = state
            delta = solve_linear(r_k, y_k)
            y_kp1 = y_k + delta
            r_kp1 = residual_fn(y_kp1)
            return (y_kp1, r_kp1, k + 1)

        def cond_fun(state):
            _, r_k, k = state
            return (jnp.linalg.norm(r_k) > self.tol) & (k < self.maxiter)

        y_final, r_final, niters = jax.lax.while_loop(cond_fun, body_fun, state0)

        def warn_callback(iters, maxiter, residual_norm, tol):
            if iters >= maxiter and residual_norm > tol:
                s1 = f"WARNING: Newton-Raphson did not converge within {int(maxiter)} iterations."
                s2 = f"Final residual norm: {float(residual_norm):.2e}."
                print(s1 + "\n" + s2)

        jax.debug.callback(
            warn_callback,
            niters, self.maxiter,
            jnp.linalg.norm(r_final), self.tol,
        )

        return y_final

"""Newton-Raphson root finder and linearisation strategies."""

import abc
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..linalg import AbstractLinearSolver, DirectDense, GMRES
from .base import AbstractRootFinder


class AbstractLineariser(eqx.Module):
    """Base class for locally linearising a system of equations."""

    @abc.abstractmethod
    def __call__(
        self, 
        fun: Callable[..., Array], 
        t: Array, 
        h: Array, 
        args: tuple
    ) -> Callable[[Array, Array], Array]:
        """Returns a solve function: (residual_at_y, y) --> delta_y"""
        raise NotImplementedError


class AutoJVP(AbstractLineariser):
    """Linearise using automatic differentiation via jax.jvp."""

    linsolver: AbstractLinearSolver

    def __init__(self, linsolver: AbstractLinearSolver = GMRES()) -> None:
        self.linsolver = linsolver

    def __call__(
        self, fun: Callable[..., Array], t: Array, h: Array, args: tuple
    ) -> Callable[[Array, Array], Array]:

        def solve(r: Array, y: Array) -> Array:

            def matvec(v: Array) -> Array:
                _, df_v = jax.jvp(lambda y_: fun(t, y_, *args), (y,), (v,))
                return v - h * df_v

            return self.linsolver(matvec, -r)

        return solve


class JVP(AbstractLineariser):
    """User-provided matrix-free Jacobian-vector product."""

    jvp_fn: Callable = eqx.field(static=True)
    linsolver: AbstractLinearSolver

    def __init__(
        self, jvp_fn: Callable, linsolver: AbstractLinearSolver = GMRES()
    ) -> None:
        self.jvp_fn = jvp_fn
        self.linsolver = linsolver

    def __call__(
        self, fun: Callable[..., Array], t: Array, h: Array, args: tuple
    ) -> Callable[[Array, Array], Array]:

        def solve(r: Array, y: Array) -> Array:
            return self.linsolver(
                lambda v: v - h * self.jvp_fn(t, y, v, *args), -r
            )

        return solve


class Jacobian(AbstractLineariser):
    """User-provided dense Jacobian."""

    jac_fn: Callable = eqx.field(static=True)
    linsolver: AbstractLinearSolver

    def __init__(
        self, jac_fn: Callable, linsolver: AbstractLinearSolver = DirectDense()
    ) -> None:
        self.jac_fn = jac_fn
        self.linsolver = linsolver

    def __call__(
        self, 
        fun: Callable[..., Array], 
        t: Array, 
        h: Array, 
        args: tuple
    ) -> Callable[[Array, Array], Array]:

        def solve(r: Array, y: Array) -> Array:
            J = jnp.eye(y.size) - h * self.jac_fn(t, y, *args)
            return self.linsolver(J, -r)

        return solve


class NewtonRaphson(AbstractRootFinder):
    """
    Newton-Raphson root-finding algorithm.

    Iterative update: y_new = y_curr - J^{-1}(y_curr) * R(y_curr)

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
    ) -> None:
        self.lineariser = lineariser
        self.tol = tol
        self.maxiter = maxiter

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
        Find the root of residual_fn(y) = 0 using Newton-Raphson method.

        Args:
            residual_fn: Residual function R(y)
            y_guess: Initial guess
            fun: Right-hand side function (passed to lineariser)
            t: Time at which to evaluate the linearisation
            h: Time step size
            args: Additional arguments to pass to fun
            theta: Implicit coefficient (see AbstractRootFinder)

        Returns:
            Solution y
        """
        solve_linear = self.lineariser(fun, t, theta * h, args)

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

        y_final, r_final, niters = jax.lax.while_loop(
            cond_fun, body_fun, state0
        )

        def warn_callback(iters, maxiter, residual_norm, tol):
            if iters >= maxiter and residual_norm > tol:
                s1 = f"WARNING: Newton-Raphson did not converge within {int(maxiter)} iterations."
                s2 = f"Final residual norm: {float(residual_norm):.2e}."
                print(s1 + "\n" + s2)

        jax.debug.callback(
            warn_callback,
            niters,
            self.maxiter,
            jnp.linalg.norm(r_final),
            self.tol,
        )

        return y_final

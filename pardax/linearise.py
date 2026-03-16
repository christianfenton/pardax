"""Linearisation methods used in the NewtonRaphson root finder.

Each lineariser bundles a linear solver and returns a solve function
with signature (residual_at_y, y) --> delta_y.
"""

import abc
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from .linsolve import AbstractLinearSolver, DirectDense, GMRES


class AbstractLineariser(eqx.Module):
    """Base class for locally linearising a system of equations."""

    @abc.abstractmethod
    def __call__(self, fun, t, h, args) -> Callable:
        """Returns a solve function: (residual_at_y, y) --> delta_y"""
        raise NotImplementedError


class AutoJVP(AbstractLineariser):
    """Linearise using automatic differentiation via jax.jvp."""

    linsolver: AbstractLinearSolver

    def __init__(self, linsolver: AbstractLinearSolver = GMRES()):
        self.linsolver = linsolver

    def __call__(self, fun, t, h, args):
        def solve(r, y):
            def matvec(v):
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
    ):
        self.jvp_fn = jvp_fn
        self.linsolver = linsolver

    def __call__(self, fun, t, h, args):
        def solve(r, y):
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
    ):
        self.jac_fn = jac_fn
        self.linsolver = linsolver

    def __call__(self, fun, t, h, args):
        def solve(r, y):
            J = jnp.eye(y.size) - h * self.jac_fn(t, y, *args)
            return self.linsolver(J, -r)

        return solve

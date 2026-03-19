"""Unit tests for the root-finding algorithms."""

import pytest

import jax.numpy as jnp

from pardax.rootfinders import NewtonRaphson, AutoJVP, JVP, Jacobian
from pardax.linalg import GMRES, DirectDense


@pytest.fixture
def backward_euler_system():
    """
    Backward Euler system for f(t, y) = 2 * y * (sqrt(2) - y).

    Residual: G(y) = y - y0 - h * f(t + h, y)

    Test that Newton-Raphson drives G(y) -> 0 with each linearisation strategy.
    """
    def fun(t, y):
        return 2.0 * y * (jnp.sqrt(2.0) - y)

    def jvp_fn(t, y, v):
        return 2.0 * (jnp.sqrt(2.0) - 2.0 * y) * v

    def jac_fn(t, y):
        return jnp.diag(2.0 * (jnp.sqrt(2.0) - 2.0 * y))

    y0 = jnp.ones((4,))
    h = 1.0
    t = 0.0

    def residual(y):
        return y - y0 - h * fun(t + h, y)

    return fun, residual, jvp_fn, jac_fn, y0, t, h


class TestRootFinders:

    def test_newton_raphson_autodiff(self, backward_euler_system):
        fun, residual, _, _, y0, t, h = backward_euler_system
        root_finder = NewtonRaphson(
            lineariser=AutoJVP(linsolver=GMRES()), tol=1e-6, maxiter=50
        )
        soln = root_finder(residual, y0, fun=fun, t=t, h=h, args=())
        assert jnp.linalg.norm(residual(soln)) < 1e-5

    def test_newton_raphson_jvp(self, backward_euler_system):
        fun, residual, jvp_fn, _, y0, t, h = backward_euler_system
        root_finder = NewtonRaphson(
            lineariser=JVP(jvp_fn=jvp_fn, linsolver=GMRES()), tol=1e-6, maxiter=50
        )
        soln = root_finder(residual, y0, fun=fun, t=t, h=h, args=())
        assert jnp.linalg.norm(residual(soln)) < 1e-5

    def test_newton_raphson_dense(self, backward_euler_system):
        fun, residual, _, jac_fn, y0, t, h = backward_euler_system
        root_finder = NewtonRaphson(
            lineariser=Jacobian(jac_fn=jac_fn, linsolver=DirectDense()),
            tol=1e-6, maxiter=50,
        )
        soln = root_finder(residual, y0, fun=fun, t=t, h=h, args=())
        assert jnp.linalg.norm(residual(soln)) < 1e-5

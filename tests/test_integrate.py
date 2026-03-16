"""Unit/integration tests for pardax.integrate."""

import pytest

import jax.numpy as jnp

from pardax.integrate import solve_ivp
from pardax.stepper import BackwardEuler
from pardax.roots import NewtonRaphson
from pardax.linearise import JVP, Jacobian
from pardax.linsolve import GMRES, DirectDense


@pytest.fixture
def simple_dynamical_system():
    """
    ODE: dy/dt = 2y (sqrt(2) - y)

    This has an attractive fixed point at y=sqrt(2) and an unstable fixed point
    at y=0.

    Initial condition: y(t=0) = 1.0
    Expected solution: y(t=t') = sqrt(2), where t' >> 0.
    """
    fun = lambda t, y: 2.0 * y * (jnp.sqrt(2.0) - y)
    jvp_fn = lambda t, y, v: 2.0 * (jnp.sqrt(2.0) - 2.0 * y) * v
    jac_fn = lambda t, y: jnp.diag(2.0 * (jnp.sqrt(2.0) - 2.0 * y))
    y0 = jnp.ones((4,))
    soln = jnp.full_like(y0, jnp.sqrt(2.0))
    return fun, jvp_fn, jac_fn, y0, soln


class TestODEIntegrator:

    def test_implicit_jvp(self, simple_dynamical_system):
        """Test implicit time stepping using a user-provided JVP."""
        fun, jvp_fn, _, y0, expected = simple_dynamical_system
        method = BackwardEuler(
            root_finder=NewtonRaphson(
                lineariser=JVP(jvp_fn=jvp_fn, linsolver=GMRES()),
                tol=1e-10,
            )
        )
        t_eval = jnp.array([0.0, 10.0])
        _, y = solve_ivp(fun, t_eval, y0, method, step_size=0.1)
        assert jnp.allclose(y[-1], expected, atol=1e-6)

    def test_implicit_autodiff(self, simple_dynamical_system):
        """Test implicit time stepping using automatic differentiation."""
        fun, _, _, y0, expected = simple_dynamical_system
        method = BackwardEuler(
            root_finder=NewtonRaphson(tol=1e-10)
        )
        t_eval = jnp.array([0.0, 10.0])
        _, y = solve_ivp(fun, t_eval, y0, method, step_size=0.1)
        assert jnp.allclose(y[-1], expected, atol=1e-6)

    def test_implicit_jac(self, simple_dynamical_system):
        """Test implicit time stepping with a dense Jacobian."""
        fun, _, jac_fn, y0, expected = simple_dynamical_system
        method = BackwardEuler(
            root_finder=NewtonRaphson(
                lineariser=Jacobian(jac_fn=jac_fn, linsolver=DirectDense()),
                tol=1e-10,
            )
        )
        t_eval = jnp.array([0.0, 10.0])
        _, y = solve_ivp(fun, t_eval, y0, method, step_size=0.1)
        assert jnp.allclose(y[-1], expected, atol=1e-6)

    def test_ode_eval(self):
        """Test solver with intermediate evaluation points."""
        y0 = jnp.array([1.0])
        t_eval = jnp.array([0.0, 1.0, 2.0])
        h = 1e-2

        t, y = solve_ivp(
            lambda t, y: -y,
            t_eval,
            y0,
            BackwardEuler(root_finder=NewtonRaphson(tol=1e-10)),
            step_size=h,
        )

        expected = jnp.exp(-t_eval)

        assert jnp.allclose(t, t_eval, atol=1e-10)
        assert jnp.allclose(y[:, 0], expected, atol=h)
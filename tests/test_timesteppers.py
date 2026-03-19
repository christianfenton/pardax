"""Unit tests for the time stepping schemes."""

import pytest

import jax
import jax.numpy as jnp

from pardax.integrate import solve_ivp
from pardax.timesteppers import ForwardEuler, BackwardEuler, RK4
from pardax.rootfinders import NewtonRaphson, AutoJVP
from pardax.linalg import GMRES


def laplacian_dirichlet_1d(
    u: jax.Array,
    bc_left: float,
    bc_right: float,
    dx: float
) -> jax.Array:
    """
    Compute the Laplacian (second derivative) using finite differences.
    Assumes ghost points at the boundaries with Dirichlet conditions.
    """
    dudx = jnp.diff(u, prepend=bc_left, append=bc_right)
    return jnp.diff(dudx) / dx**2


def heat_rhs_dirichlet(
    t: float,
    u: jax.Array,
    diffusivity: float,
    bc_left: float,
    bc_right: float,
    dx: float,
) -> jax.Array:
    """Right-hand side of heat equation: du/dt = D d²u/dx²"""
    d2udx2 = laplacian_dirichlet_1d(u, bc_left, bc_right, dx)
    return diffusivity * d2udx2


def gaussian_ic(x: jax.Array, t: float, D: float, L: float) -> jax.Array:
    """Analytical Gaussian solution to the heat equation."""
    k = 1.0 / jnp.sqrt(4.0 * jnp.pi * D * t)
    return k * jnp.exp(-((x - L / 2.0)**2) / (4.0 * D * t))


@pytest.fixture
def heat_equation_setup():
    """
    Setup for 1D heat equation with Dirichlet BCs.

    Problem: du/dt = D d²u/dx² on [0, L] with u(0,t) = u(L,t) = 0
    Initial condition: Gaussian centered at L/2
    """
    D = 2.0
    L = 100.0
    n = 32
    dx = L / (n + 1)
    bc_values = (0.0, 0.0)

    x = jnp.linspace(dx, L - dx, n, endpoint=True)

    return {
        'D': D,
        'L': L,
        'n': n,
        'dx': dx,
        'x': x,
        'bc_values': bc_values,
    }


class TestExplicitMethods:
    """Test explicit time-stepping methods on the heat equation."""

    def test_forward_euler(self, heat_equation_setup):
        setup = heat_equation_setup
        D = setup['D']
        L = setup['L']
        dx = setup['dx']
        x = setup['x']
        bc_left, bc_right = setup['bc_values']

        t_start = 1.0
        t_end = 1.5

        cfl = 0.2
        dt = cfl * dx**2 / D

        y0 = gaussian_ic(x, t_start, D, L)

        method = ForwardEuler()
        t_eval = jnp.array([t_start, t_end])
        _, y = solve_ivp(
            heat_rhs_dirichlet,
            t_eval,
            y0,
            method,
            dt_max=dt,
            args=(D, bc_left, bc_right, dx)
        )

        y_exact = gaussian_ic(x, t_end, D, L)

        assert jnp.allclose(y[-1], y_exact, atol=dt)

    def test_rk4(self, heat_equation_setup):
        setup = heat_equation_setup
        D = setup['D']
        L = setup['L']
        dx = setup['dx']
        x = setup['x']
        bc_left, bc_right = setup['bc_values']

        t_start = 1.0
        t_end = 5.0

        cfl = 0.2
        dt = cfl * dx**2 / D

        y0 = gaussian_ic(x, t_start, D, L)

        method = RK4()
        t_eval = jnp.array([t_start, t_end])
        _, y = solve_ivp(
            heat_rhs_dirichlet,
            t_eval,
            y0,
            method,
            dt_max=dt,
            args=(D, bc_left, bc_right, dx)
        )

        y_exact = gaussian_ic(x, t_end, D, L)

        assert jnp.allclose(y[-1], y_exact, atol=dt**2)


class TestImplicitMethods:
    """Test implicit time-stepping methods on the heat equation."""

    def test_backward_euler_gmres(self, heat_equation_setup):
        setup = heat_equation_setup
        D = setup['D']
        L = setup['L']
        dx = setup['dx']
        x = setup['x']
        bc_left, bc_right = setup['bc_values']

        t_start = 1.0
        t_end = 5.0

        cfl = 1.5
        dt = cfl * dx**2 / D

        y0 = gaussian_ic(x, t_start, D, L)

        linsolver = GMRES(tol=1e-8, maxiter=100)
        solver = NewtonRaphson(
            lineariser=AutoJVP(linsolver=linsolver), tol=1e-8, maxiter=20
        )
        method = BackwardEuler(root_finder=solver)

        t_eval = jnp.array([t_start, t_end])
        _, y = solve_ivp(
            heat_rhs_dirichlet,
            t_eval,
            y0,
            method,
            dt_max=dt,
            args=(D, bc_left, bc_right, dx)
        )

        y_exact = gaussian_ic(x, t_end, D, L)

        assert jnp.allclose(y[-1], y_exact, atol=dt**2)

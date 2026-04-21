"""Unit tests for the time stepping schemes."""

import pytest

import jax.numpy as jnp

from pardax.ivp import solve_ivp, integrate
from pardax.timesteppers import ForwardEuler, BackwardEuler, RK4
from pardax.rootfinders import NewtonRaphson, AutoJVP, LinearRootFinder
from pardax.linalg import GMRES, SpectralOperator, SpectralSolver
from pardax.transform import dst1, idst1


def laplacian_dirichlet_1d(u, bc_left, bc_right, dx):
    dudx = jnp.diff(u, prepend=bc_left, append=bc_right)
    return jnp.diff(dudx) / dx**2


def heat_rhs(t, u, params):
    return params["D"] * laplacian_dirichlet_1d(
        u, params["bc_left"], params["bc_right"], params["dx"]
    )


def gaussian(x, t, diffusivity, length):
    return jnp.exp(
        -((x - length / 2) ** 2) / (4 * diffusivity * t)
    ) / jnp.sqrt(4 * jnp.pi * diffusivity * t)


@pytest.fixture
def heat_setup():
    """Setup for 1D heat equation with Dirichlet BCs."""
    diffusivity = 2.0
    length = 50.0
    n = 128
    dx = length / (n + 1)
    bc_left, bc_right = 0.0, 0.0
    x = jnp.linspace(dx, length - dx, n, endpoint=True)
    return {
        "D": diffusivity,
        "L": length,
        "dx": dx,
        "x": x,
        "bc_left": bc_left,
        "bc_right": bc_right,
    }


class TestForwardEuler:
    @pytest.fixture
    def stepper(self):
        return ForwardEuler()

    @pytest.fixture
    def step_size(self, heat_setup):
        return 0.5 * heat_setup["dx"] ** 2 / heat_setup["D"]

    @pytest.fixture
    def atol(self, step_size, heat_setup):
        return max(step_size, heat_setup["dx"] ** 2)

    def test_solve_ivp(self, heat_setup, stepper, step_size, atol):
        diffusivity, length = heat_setup["D"], heat_setup["L"]
        x = heat_setup["x"]

        t_span = (1.0, 5.0)
        y0 = gaussian(x, t_span[0], diffusivity, length)
        params = {k: heat_setup[k] for k in ("D", "dx", "bc_left", "bc_right")}

        t, y = solve_ivp(
            heat_rhs,
            t_span=t_span,
            y0=y0,
            stepper=stepper,
            step_size=step_size,
            params=params,
            num_checkpoints=2,
        )

        for i in range(len(t)):
            assert jnp.allclose(
                y[i], gaussian(x, t[i], diffusivity, length), atol=atol
            )

    def test_adaptive_solve(self, heat_setup, stepper, step_size, atol):
        diffusivity, length = heat_setup["D"], heat_setup["L"]
        x = heat_setup["x"]

        t_eval = jnp.linspace(1.0, 5.0, 4)
        y0 = gaussian(x, t_eval[0], diffusivity, length)
        params = {k: heat_setup[k] for k in ("D", "dx", "bc_left", "bc_right")}

        def step_size_fn(t, u, params):
            return step_size

        t, y = integrate(
            heat_rhs,
            t_eval=t_eval,
            y0=y0,
            stepper=stepper,
            step_size_fn=step_size_fn,
            params=params,
        )

        for i in range(len(t)):
            soln = gaussian(x, t[i], diffusivity, length)
            assert jnp.allclose(y[i], soln, atol=atol)


class TestRK4:
    @pytest.fixture
    def stepper(self):
        return RK4()

    @pytest.fixture
    def step_size(self, heat_setup):
        return 0.5 * heat_setup["dx"] ** 2 / heat_setup["D"]

    @pytest.fixture
    def atol(self, step_size, heat_setup):
        return max(step_size**4, heat_setup["dx"] ** 2)

    def test_solve_ivp(self, heat_setup, stepper, step_size, atol):
        diffusivity, length = heat_setup["D"], heat_setup["L"]
        x = heat_setup["x"]

        t_span = (1.0, 5.0)
        y0 = gaussian(x, t_span[0], diffusivity, length)
        params = {k: heat_setup[k] for k in ("D", "dx", "bc_left", "bc_right")}

        t, y = solve_ivp(
            heat_rhs,
            t_span=t_span,
            y0=y0,
            stepper=stepper,
            step_size=step_size,
            params=params,
            num_checkpoints=2,
        )

        for i in range(len(t)):
            soln = gaussian(x, t[i], diffusivity, length)
            assert jnp.allclose(y[i], soln, atol=atol)

    def test_adaptive_solve(self, heat_setup, stepper, step_size, atol):
        diffusivity, length = heat_setup["D"], heat_setup["L"]
        x = heat_setup["x"]

        t_eval = jnp.linspace(1.0, 5.0, 4)
        y0 = gaussian(x, t_eval[0], diffusivity, length)
        params = {k: heat_setup[k] for k in ("D", "dx", "bc_left", "bc_right")}

        def step_size_fn(t, u, params):
            return step_size

        t, y = integrate(
            heat_rhs,
            t_eval=t_eval,
            y0=y0,
            stepper=stepper,
            step_size_fn=step_size_fn,
            params=params,
        )

        for i in range(len(t)):
            soln = gaussian(x, t[i], diffusivity, length)
            assert jnp.allclose(y[i], soln, atol=atol)


class TestBackwardEuler:
    """Tests for the Backward Euler time stepper."""

    @pytest.fixture
    def stepper(self):
        linsolver = GMRES(tol=1e-8, maxiter=100)
        root_finder = NewtonRaphson(
            lineariser=AutoJVP(linsolver=linsolver), tol=1e-8, maxiter=20
        )
        return BackwardEuler(root_finder=root_finder)

    @pytest.fixture
    def step_size(self, heat_setup):
        return 0.5 * heat_setup["dx"] ** 2 / heat_setup["D"]

    @pytest.fixture
    def atol(self, step_size, heat_setup):
        return max(step_size, heat_setup["dx"] ** 2)

    def test_solve_ivp(self, heat_setup, stepper, step_size, atol):
        diffusivity, length = heat_setup["D"], heat_setup["L"]
        x = heat_setup["x"]

        t_span = (1.0, 5.0)
        y0 = gaussian(x, t_span[0], diffusivity, length)
        params = {k: heat_setup[k] for k in ("D", "dx", "bc_left", "bc_right")}

        t, y = solve_ivp(
            heat_rhs,
            t_span=t_span,
            y0=y0,
            stepper=stepper,
            step_size=step_size,
            params=params,
            num_checkpoints=2,
        )

        for i in range(len(t)):
            soln = gaussian(x, t[i], diffusivity, length)
            assert jnp.allclose(y[i], soln, atol=atol)

    def test_adaptive_solve(self, heat_setup, stepper, step_size, atol):
        diffusivity, length = heat_setup["D"], heat_setup["L"]
        x = heat_setup["x"]

        t_eval = jnp.linspace(1.0, 5.0, 4)
        y0 = gaussian(x, t_eval[0], diffusivity, length)
        params = {k: heat_setup[k] for k in ("D", "dx", "bc_left", "bc_right")}

        def step_size_fn(t, u, params):
            return step_size

        t, y = integrate(
            heat_rhs,
            t_eval=t_eval,
            y0=y0,
            stepper=stepper,
            step_size_fn=step_size_fn,
            params=params,
        )

        for i in range(len(t)):
            soln = gaussian(x, t[i], diffusivity, length)
            assert jnp.allclose(y[i], soln, atol=atol)


class TestSpectralBackwardEuler:
    """Backward Euler with a spectral (DST) linear solver."""

    @pytest.fixture
    def stepper(self, heat_setup):
        diffusivity = heat_setup["D"]
        dx = heat_setup["dx"]
        n = len(heat_setup["x"])

        # Eigenvalues of the 2nd-order Laplacian with Dirichlet BCs
        k = jnp.arange(1, n + 1)
        sigma = (
            -4 * diffusivity / dx**2 * jnp.sin(k * jnp.pi / (2 * (n + 1))) ** 2
        )

        operator = SpectralOperator(eigvals=sigma)
        linsolver = SpectralSolver(forward=dst1, backward=idst1)
        root_finder = LinearRootFinder(linsolver=linsolver, operator=operator)
        return BackwardEuler(root_finder=root_finder)

    @pytest.fixture
    def step_size(self, heat_setup):
        return 0.5 * heat_setup["dx"] ** 2 / heat_setup["D"]

    @pytest.fixture
    def atol(self, step_size):
        return step_size

    def test_solve_ivp(self, heat_setup, stepper, step_size, atol):
        diffusivity, length = heat_setup["D"], heat_setup["L"]
        x = heat_setup["x"]

        t_span = (1.0, 5.0)
        y0 = gaussian(x, t_span[0], diffusivity, length)
        params = {k: heat_setup[k] for k in ("D", "dx", "bc_left", "bc_right")}

        t, y = solve_ivp(
            heat_rhs,
            t_span=t_span,
            y0=y0,
            stepper=stepper,
            step_size=step_size,
            params=params,
            num_checkpoints=2,
        )

        for i in range(len(t)):
            assert jnp.allclose(
                y[i], gaussian(x, t[i], diffusivity, length), atol=atol
            )

    def test_adaptive_solve(self, heat_setup, stepper, step_size, atol):
        diffusivity, length = heat_setup["D"], heat_setup["L"]
        x = heat_setup["x"]

        t_eval = jnp.linspace(1.0, 5.0, 4)
        y0 = gaussian(x, t_eval[0], diffusivity, length)
        params = {k: heat_setup[k] for k in ("D", "dx", "bc_left", "bc_right")}

        def step_size_fn(t, u, params):
            return step_size

        t, y = integrate(
            heat_rhs,
            t_eval=t_eval,
            y0=y0,
            stepper=stepper,
            step_size_fn=step_size_fn,
            params=params,
        )

        for i in range(len(t)):
            assert jnp.allclose(
                y[i], gaussian(x, t[i], diffusivity, length), atol=atol
            )

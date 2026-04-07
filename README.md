# pardax

A JAX-native solver for initial value problems.

[![tests](https://github.com/christianfenton/pardax/actions/workflows/tests.yml/badge.svg)](https://github.com/christianfenton/pardax/actions/workflows/tests.yml)
[![lint](https://github.com/christianfenton/pardax/actions/workflows/lint.yml/badge.svg)](https://github.com/christianfenton/pardax/actions/workflows/lint.yml)
[![docs](https://github.com/christianfenton/pardax/actions/workflows/docs.yml/badge.svg)](https://github.com/christianfenton/pardax/actions/workflows/docs.yml)

`pardax` provides a variety of composable time-stepping schemes that 
work seamlessly with JAX transformations, along with a familiar interface 
inspired by [`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)

Check out the [documentation](https://christianfenton.github.io/pardax).

## Installation

[uv](https://docs.astral.sh/uv/) is recommended for installation.

### Using uv

With SSH:
```bash
uv init my-project
cd my-project
uv add git+ssh://git@github.com/christianfenton/pardax.git
```

With HTTPS:
```bash
uv add git+https://github.com/christianfenton/pardax.git
```

### Using pip

With SSH:
```bash
pip install git+ssh://git@github.com/christianfenton/pardax.git
```

With HTTPS:
```bash
pip install git+https://github.com/christianfenton/pardax.git
```

## Quick start

```python
import jax.numpy as jnp
import pardax as pdx

# 1. Define your discretised PDE as an ODE
# NOTE: must be functionally pure (JAX-compatible)
def my_pde_rhs(t, y, *args):
    """dy/dt = f(t, y, ...)"""
    ...

# 2. Set initial condition
y0 = ...

# 3. Choose time-stepping method
method = pdx.RK4()

# 4. Integrate
t, y = pdx.solve_ivp(
    my_pde_rhs,
    t_eval=jnp.linspace(0.0, 1.0, 100),
    y0=y0,
    stepper=method,
    dt_max=0.001,
    args=(...,)
)
```

## Development

Dependencies are split into the following groups:

| Group | Purpose | Install |
|-------|---------|---------|
| `test` | pytest, beartype | `uv sync --group test` |
| `docs` | mkdocs and plugins | `uv sync --group docs` |
| `lint` | ruff | `uv sync --group lint` |
| `notebooks` | matplotlib, ipykernel | `uv sync --extra notebooks` |

To install everything at once:
```bash
uv sync --all-groups --all-extras
```

During development, it might be helpful to serve the documentation locally
by running:
```bash
uv sync --group docs
uv run mkdocs serve
```
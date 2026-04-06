# pardax

A JAX-native solver for initial value problems.

![Coverage](coverage.svg)

`pardax` provides a variety of composable time-stepping schemes that 
work seamlessly with JAX transformations, along with a familiar interface 
inspired by [`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)

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

## Documentation

Build the documentation cloning the repository and running
```bash
uv sync --group docs
uv run mkdocs build
```
or serve them as a local webpage with
```bash
uv run mkdocs serve
```

## Development

Dependencies are split into the following groups:

| Group | Purpose | Install |
|-------|---------|---------|
| `test` | pytest, beartype | `uv sync --group test` |
| `docs` | mkdocs and plugins | `uv sync --group docs` |
| `notebooks` | matplotlib, ipykernel | `uv sync --extra notebooks` |
| `lint` | ruff, mypy | `uv sync --group lint` |

To install everything at once:
```bash
uv sync --all-groups --all-extras
```
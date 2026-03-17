# pardax

A flexible solver interface for partial differential equations, written in JAX.

## Installation

[uv](https://docs.astral.sh/uv/) is recommended for installation.

### Using uv

With SSH:
```bash
uv new my-project
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
# NOTE: This must be written in a JAX-compatible (functionally pure) way
def my_pde_rhs(t, y, *args):
    """Right-hand side: dy/dt = f(t, y, ...)

    Implement your spatial discretisation here.
    Handle boundary conditions within this function.
    """
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

Build the documentation by running
```bash
uv run mkdocs build
```
or serve them as a local webpage with
```bash
uv run mkdocs serve
```
# pardax

A JAX-native solver for initial value problems.

[![tests](https://github.com/christianfenton/pardax/actions/workflows/tests.yml/badge.svg)](https://github.com/christianfenton/pardax/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/christianfenton/pardax/graph/badge.svg)](https://codecov.io/gh/christianfenton/pardax)
[![lint](https://github.com/christianfenton/pardax/actions/workflows/lint.yml/badge.svg)](https://github.com/christianfenton/pardax/actions/workflows/lint.yml)
[![docs](https://github.com/christianfenton/pardax/actions/workflows/docs.yml/badge.svg)](https://github.com/christianfenton/pardax/actions/workflows/docs.yml)

`pardax` provides a variety of composable time-stepping schemes that 
work seamlessly with JAX transformations, along with a familiar interface 
inspired by [`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)

Check out the [documentation page](https://christianfenton.github.io/pardax)
for more details.

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

```python notest
import pardax as pdx

# 1. Define your ODE
# NOTE: must be functionally pure (JAX-compatible)
def my_pde_fun(t: float, y: jax.Array, params: PyTree):
    """dy/dt = f(t, y, ...)"""
    ...

# 2. Set initial condition
y0 = ...

# 3. Choose time-stepping method
method = pdx.RK4()

# 4. Integrate
t, y = pdx.solve_ivp(
    my_pde_fun,
    t_span=(0.0, 1.0),
    y0=y0,
    stepper=method,
    step_size=0.001,
    params=params
)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on setting up a 
development environment, running tests, linting, and building the documentation
locally.
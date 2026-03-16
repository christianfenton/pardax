# pardax

A flexible solver interface for partial differential equations, written in JAX.

## Installation

[Poetry](https://python-poetry.org/docs/) is recommended for installation.

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

## Tutorials

To get started using the package, check out the tutorials:

- **[Solving the heat equation](tutorials/heat_equation.md)**

## Links

Check out the source code on [GitHub](https://github.com/christianfenton/pardax).

## Future Works

In the future, `pardax.integrate.solve_ivp` should be adapted to match 
[`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
and added to [`jax.scipy.integrate`](https://docs.jax.dev/en/latest/jax.scipy.html#module-jax.scipy.integrate).
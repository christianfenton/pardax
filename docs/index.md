# pardax

A JAX-native solver for initial value problems.

`pardax` provides a variety of composable time-stepping schemes that 
work seamlessly with most JAX transformations, and a familiar interface 
inspired by [`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)

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

## Next steps

- [Getting started](guide.md)
- [Heat equation with backward Euler](tutorials/implicit_heat.md)
- [Burgers' equation with a pseudo-spectral method](tutorials/spectral_burgers.md)
- [Extending the solver](extending.md)
- [API reference](api.md)

Read the source code on
[GitHub](https://github.com/christianfenton/pardax).
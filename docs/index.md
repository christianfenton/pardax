# pardax

A JAX-native solver for initial value problems.

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

## Next steps

- [Getting started](guide.md)
- [Heat equation with backward Euler](tutorials/implicit_heat.md)
- [Burgers' equation with a pseudo-spectral method](tutorials/spectral_burgers.md)
- [Extending the solver](extending.md)
- [API reference](api.md)

Read the source code on
[GitHub](https://github.com/christianfenton/pardax).

## FAQ

### *How does pardax compare with established libraries like diffrax?*

[diffrax](https://docs.kidger.site/diffrax/) is a larger, more general JAX-based 
differential equation library, while `pardax` is intentionally minimal and 
modelled on `scipy.integrate.solve_ivp`.

Some of the main differences between these two libraries are:

- Scope: `diffrax` supports ODEs, SDEs (stochastic), and CDEs (controlled),
    while `pardax` focuses on ODEs and PDEs only
- Solvers: `diffrax` currently offers a larger catalogue of solvers than `pardax`, 
    however both are composable and designed to support user-defined solvers
- Stepping: `diffrax` has an adaptive step size controller, while `pardax` uses a fixed step size or a user-supplied state-dependent callback (e.g. a CFL condition)
- Spectral methods: `pardax` has built-in support for spectral methods, which
    `diffrax` doesn't have at the time of writing

If you need the broader feature set that `diffrax` offers, it is likely the better choice.
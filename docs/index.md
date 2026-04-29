# pardax

A JAX-native solver for initial value problems.

`pardax` provides a variety of composable time-stepping schemes that 
work seamlessly with JAX transformations, along with a familiar interface 
inspired by [`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)

## Installation

```bash
pip install pardax
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv add pardax
```

## Next steps

- [Getting started](guide.md)
- [Tutorial: Heat equation with backward Euler](tutorials/implicit_heat.md)
- [Tutorial: Burgers' equation with a pseudo-spectral method](tutorials/spectral_burgers.md)
- [Extending the solver](extending.md)
- [API reference](api.md)

Read the source code on [GitHub](https://github.com/christianfenton/pardax).
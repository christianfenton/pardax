# FAQ

## *How does pardax compare with established libraries like diffrax?*

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
# Getting started

`pardax` is a JAX-native ODE integrator for solving initial value
problems. It is fully compatible with JAX transformations (`jit`,
`vmap`, `grad`), composable, and designed for the semi-discrete
approach to PDEs: users handle the spatial discretisation, and `pardax`
handles the time integration.

## Quick start

```python
import jax.numpy as jnp
import pardax as pdx

# 1. Define your discretised PDE as an ODE
# NOTE: must be functionally pure (JAX-compatible)
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
    t_span=(0.0, 1.0),
    y0=y0,
    stepper=method,
    step_size=0.001,
    args=(...,),
    num_checkpoints=10,
)
```

## Choosing a solver

`pardax` provides two top-level integration functions.

[solve_ivp][pardax.solve_ivp] divides `t_span` into equally-spaced
segments and saves a snapshot at the end of each one. It uses
`jax.lax.scan` internally, which makes it compatible with all JAX
transformations including reverse-mode automatic differentiation.

[integrate][pardax.integrate] advances the solution to an arbitrary
sequence of times in `t_eval`, using a step-size callback that can
depend on the current solution state. It uses `jax.lax.while_loop`
internally, so it supports `jax.jit` and `jax.vmap` but only forward-mode
automatic differentation. Use this when you need CFL-based or otherwise 
state-dependent step sizes, or when output times are not equally spaced.

```python
def cfl_step_size(t, u, nu, dx):
    return 0.5 * dx**2 / nu

t, y = pdx.integrate(
    fun,
    t_eval=jnp.linspace(0.0, 5.0, 11),
    y0=y0,
    stepper=pdx.RK4(),
    step_size_fn=cfl_step_size,
    args=(nu, dx),
)
```

## JAX transformations

Because `pardax` is built on JAX and [Equinox](https://docs.kidger.site/equinox/),
you can apply JAX transformations directly to `solve_ivp`.

### JIT compilation

Compile the entire integration for faster execution:

```python
import jax

solve_jit = jax.jit(lambda y_: pdx.solve_ivp(
    fun, t_span, y_, stepper, step_size, args
))

t, y = solve_jit(y0)
```

### Vectorisation

Integrate multiple initial conditions in parallel with `vmap`:

```python
y0_batch = jnp.stack([y0_1, y0_2, y0_3])  # (batch, n)

solve_batch = jax.vmap(
    lambda y_: pdx.solve_ivp(fun, t_span, y_, stepper, step_size, args)
)

t, y_batch = solve_batch(y0_batch)
```

### Differentiation

Differentiate through the solver for sensitivity analysis or parameter
optimisation:

```python
def loss(params):
    t, y = pdx.solve_ivp(fun, t_span, y0, stepper, step_size, args=(params,))
    return jnp.mean((y[-1] - y_target)**2)

grads = jax.grad(loss)(params)
```

## Time-stepping methods

### Explicit methods

Explicit methods compute the next state directly from the current
state. They are simple and efficient per step but require small time
steps for stiff problems.

```python
method = pdx.ForwardEuler()   # first-order
method = pdx.RK4()            # fourth-order
```

### Implicit methods

Implicit methods solve a non-linear or linear system at each time step.
They are more expensive per step but can take much larger time steps for
stiff problems (e.g. diffusion-dominated PDEs).

The implicit solver is assembled from composable components:

```python
# Linear solver -> Lineariser -> Root finder -> Time stepper
method = pdx.BackwardEuler(
    root_finder=pdx.NewtonRaphson(
        lineariser=pdx.AutoJVP(linsolver=pdx.GMRES()),
        tol=1e-6,
    )
)
```

For linear problems, you can skip Newton iteration entirely and solve
the implicit system in a single step using a
[LinearRootFinder][pardax.LinearRootFinder]. See the
[Burgers' equation tutorial](tutorials/spectral_burgers.md) for an
example.

### IMEX (implicit-explicit) splitting

When a problem has both stiff and non-stiff terms, an IMEX scheme
treats each with an appropriate method:

```python
rhs = {
    "implicit": stiff_term,      # e.g. diffusion
    "explicit": non_stiff_term,  # e.g. advection
}

method = pdx.IMEX(
    implicit=pdx.BackwardEuler(root_finder=...),
    explicit=pdx.RK4(),
)

t, y = pdx.solve_ivp(rhs, t_span, y0, method, step_size=dt, args=args)
```

This eliminates the stiff stability constraint while keeping the
non-stiff part cheap. See the
[Burgers' equation tutorial](tutorials/spectral_burgers.md) for a
complete example.
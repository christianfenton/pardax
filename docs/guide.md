# Getting started

The `pardax.integrate` module provides JAX-compatible time integration methods
for solving initial value problems (IVPs). The solver performs temporal
discretisation and integration, while users need to handle spatial
discretisation during setup.

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

## Time-stepping methods

### Explicit methods

Explicit methods compute the next state directly from the current state.
They're simple and fast but can require very small time-steps
for stiff problems.

### Implicit methods

Implicit methods use a root-finding algorithm at each time step, and
root-finding algorithms often use linear solvers at each iteration.
Implicit methods are usually more expensive than explicit methods per step but
can allow much larger time steps for stiff problems.

### JAX transformations

As long as `fun` is JAX-compatible,
[solve_ivp][pardax.solve_ivp] should support most JAX transformations.

JIT Compilation:

```python
import jax
import pardax as pdx

# JIT-compile the entire integration
solve_jit = jax.jit(
    pdx.solve_ivp(fun, t_eval, y0, stepper, dt_max, args),
    static_argnames=['fun']
)

t_final, y_final = solve_jit(y0)
```

Vectorisation (batching):

```python
# Integrate multiple initial conditions in parallel
y0_batch = jnp.stack([y0_1, y0_2, y0_3])  # (batch, n)

solve_batch = jax.vmap(
    lambda y_: pdx.solve_ivp(fun, t_eval, y_, stepper, dt_max, args)[1]
)

y_final_batch = solve_batch(y0_batch)  # (batch, n)
```

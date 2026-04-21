# Getting started

In `pardax`, users are expected to handle the spatial discretisation of
their ODE, and `pardax` handles the rest.

```python notest
import jax.numpy as jnp
import pardax as pdx

# 1. Define your discretised PDE as an ODE
# NOTE: must be functionally pure (JAX-compatible)
def my_pde_rhs(t, y, params):
    """Right-hand side: dy/dt = f(t, y, params)

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
    params={...},
    num_checkpoints=10,
)
```

`pardax` also provides [integrate][pardax.integrate], which takes a step-size 
callback that can depend on the current solution state. It uses 
`jax.lax.while_loop` internally, so does not support reverse-mode automatic
differentiation with `jax.grad`.

```python notest
def cfl_step_size(t, u, params):
    return 0.5 * params["dx"]**2 / params["nu"]

t, y = pdx.integrate(
    fun,
    t_eval=jnp.linspace(0.0, 5.0, 11),
    y0=y0,
    stepper=pdx.RK4(),
    step_size_fn=cfl_step_size,
    params={"nu": nu, "dx": dx},
)
```

## JAX transformations

Because `pardax` is built on JAX and [Equinox](https://docs.kidger.site/equinox/),
you can apply JAX transformations directly to `solve_ivp`.

### Vectorisation

```python notest
y0_batch = jnp.stack([y0_1, y0_2, y0_3])  # (batch, n)

solve_batch = jax.vmap(
    lambda y_: pdx.solve_ivp(fun, t_span, y_, stepper, step_size, params)
)

t, y_batch = solve_batch(y0_batch)
```

### Differentiation

```python notest
def loss(params):
    t, y = pdx.solve_ivp(fun, t_span, y0, stepper, step_size, params=params)
    return jnp.mean((y[-1] - y_target)**2)

grads = jax.grad(loss)(params)
```

### JIT compilation

```python notest
import jax

solve_jit = jax.jit(lambda y_: pdx.solve_ivp(
    fun, t_span, y_, stepper, step_size, params
))

t, y = solve_jit(y0)
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

### Advanced methods

`pardax` is designed so that time-stepping schemes are composable,
which allows users to implement their own schemes that treat PDE terms
separately. See [Extending the solver](extending.md) for 
more information or the 
[Burgers' equation tutorial](tutorials/spectral_burgers.md) for a
worked example.
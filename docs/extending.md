# Extending the solver

All of the core components in `pardax` (time steppers, root finders,
linear solvers, and linear operators) can be extended by subclassing
the relevant abstract base class.

## Custom time-stepping methods

### Subclassing `AbstractStepper`

The simplest way to add a new time-stepping method is to subclass
[AbstractStepper][pardax.AbstractStepper] and implement `__call__`.
This is the recommended approach when your method takes a single
right-hand side function with the standard `fun(t, y, params) -> dy/dt`
signature:

```python notest
import pardax as pdx

class Heun(pdx.AbstractStepper):

    def __call__(self, fun, t, y, step_size, params=None):
        """Heun's method."""
        k1 = fun(t, y, params)
        k2 = fun(t + step_size, y + step_size * k1, params)
        return y + 0.5 * step_size * (k1 + k2), self

t, y = pdx.solve_ivp(fun, t_span, y0, Heun(), step_size=0.01, params=params)
```

### Duck typing with `StepperLike`

[solve_ivp][pardax.solve_ivp] accepts any object that satisfies the
[StepperLike][pardax.StepperLike] protocol - that is, any object with a
`__call__(self, fun, t, y, step_size, params=None)` method that returns
`(y_new, updated_stepper)`. This does not need to inherit from `AbstractStepper`.

This is useful when:

- The stepper carries internal state between steps (e.g. a multi-step method)
- Wrapping an external library

```python notest
import equinox as eqx
import jax

class AdamsBashforth2(eqx.Module):
    """Two-step Adams-Bashforth. Initialise f_prev with zeros or a warm-up step."""

    f_prev: jax.Array

    def __call__(self, fun, t, y, step_size, params=None):
        f_curr = fun(t, y, params)
        y_new = y + step_size * (1.5 * f_curr - 0.5 * self.f_prev)
        return y_new, eqx.tree_at(lambda s: s.f_prev, self, f_curr)

stepper = AdamsBashforth2(f_prev=jnp.zeros_like(y0))
t, y = pdx.solve_ivp(fun, t_span, y0, stepper, step_size=0.01, params=params)
```

### IMEX splitting

An implicit-explicit (IMEX) scheme treats stiff and non-stiff parts of the
right-hand side separately. Because `solve_ivp` and `integrate` expect a
single callable as `fun`, IMEX schemes requires writing a custom 
time-stepping loop directly with `jax.lax.scan` or `jax.lax.while_loop`. 
Each sub-stepper receives its own callable, and the updated stepper state 
is carried through the loop.

See the [Burgers' equation tutorial](tutorials/spectral_burgers.md) for a
worked example of a pseudo-spectral IMEX scheme.

## Custom root finders

Root finders are used by implicit time steppers (e.g.
[BackwardEuler][pardax.BackwardEuler]) to solve the nonlinear or
linear system that arises at each time step.

The built-in root finders are:

- [NewtonRaphson][pardax.NewtonRaphson] -- iterative, for nonlinear systems.
- [LinearRootFinder][pardax.LinearRootFinder] -- single-step, for linear systems.

Users can add new root finders by subclassing
[AbstractRootFinder][pardax.AbstractRootFinder].

## Custom linear solvers

Linear solvers are used as subroutines inside root finders and
linearisers.

The linear solvers currently available are:

- [GMRES][pardax.GMRES]
- [CG][pardax.CG]
- [BiCGStab][pardax.BiCGStab]
- [DirectDense][pardax.DirectDense]
- [SpectralSolver][pardax.SpectralSolver]

Users can add new solvers by subclassing
[AbstractLinearSolver][pardax.AbstractLinearSolver].

## Custom linear operators

Linear operators build the implicit system `(I - step_size * L)` for use with
[LinearRootFinder][pardax.LinearRootFinder]. Each operator returns the
system in the form expected by its paired linear solver.

The built-in operators are:

- [DenseOperator][pardax.DenseOperator] -- returns a dense matrix.
  Pair with [DirectDense][pardax.DirectDense].
- [MatrixFreeOperator][pardax.MatrixFreeOperator] -- returns a matvec
  callable. Pair with an iterative solver
  ([GMRES][pardax.GMRES], [CG][pardax.CG], [BiCGStab][pardax.BiCGStab]).
- [SpectralOperator][pardax.SpectralOperator] -- returns a spectral
  symbol array. Pair with [SpectralSolver][pardax.SpectralSolver].

Users can add new operators by subclassing
[AbstractLinearOperator][pardax.AbstractLinearOperator].

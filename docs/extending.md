# Extending the solver

All of the core components in `pardax` (time steppers, root finders,
linear solvers, and linear operators) can be extended by subclassing
the relevant abstract base class.

## Custom time-stepping methods

### Subclassing `AbstractStepper`

The simplest way to add a new time-stepping method is to subclass
[AbstractStepper][pardax.AbstractStepper] and implement the `step`
method. This is the recommended approach when your method takes
a single right-hand side function with the standard
`(t, y, *args) -> dy/dt` signature:

```python notest
import equinox as eqx
from jax import Array
import pardax as pdx

class Heun(pdx.AbstractStepper):

    def step(self, fun, t, y, h, args=()):
        """Heun's method."""
        k1 = fun(t, y, *args)
        k2 = fun(t + h, y + h * k1, *args)
        return y + 0.5 * h * (k1 + k2)

t, y = pdx.solve_ivp(fun, t_span, y0, Heun(), step_size=0.01, args=args)
```

### Duck typing with `StepperLike`

[solve_ivp][pardax.solve_ivp] accepts any object that satisfies the
[StepperLike][pardax.StepperLike] protocol -- that is, any object with
a `step(self, fun, t, y, h, args=())` method that returns an `Array`.
You do not need to inherit from `AbstractStepper`.

This is useful when:

- Your stepper needs a different kind of right-hand side (e.g. a dict,
  a named tuple, or a custom object).
- You want to compose multiple sub-steppers, as
  [IMEX][pardax.IMEX] does.
- You are wrapping an external library.

```python notest
import equinox as eqx
from jax import Array

class MyProjectionStepper(eqx.Module):
    """A stepper whose `fun` returns (tendency, projection) pairs."""

    def step(self, fun, t, y, h, args=()):
        tendency, project = fun(t, y, *args)
        return project(y + h * tendency)

t, y = pdx.solve_ivp(fun, t_span, y0, MyProjectionStepper(), step_size=0.01)
```

The built-in [IMEX][pardax.IMEX] stepper is an example of this
pattern: it accepts a `dict` with `"implicit"` and `"explicit"` keys
and delegates each part to a separate `AbstractStepper`.

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

Linear operators build the implicit system `(I - h * L)` for use with
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

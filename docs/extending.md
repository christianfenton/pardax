# Extending the solver

All of the core components in `pardax` (time steppers, root finders, and
linear solvers) can be extended by subclassing the relevant abstract base
class.

## Custom time-stepping methods

Subclass [AbstractStepper][pardax.AbstractStepper] and implement the
`step` method:

```python
from jax import Array
import pardax as pdx

class MyMethod(pdx.AbstractStepper):

    def step(
        self,
        fun: pdx.RHS,  # pdx.RHS is a type alias for Callable[..., Array]
        t: Array,
        y: Array,
        h: Array,
        args: tuple = ()
    ) -> Array:
        """Advance one time step."""
        ...

t, y = pdx.solve_ivp(fun, t_eval, y0, MyMethod(), dt_max, args)
```

## Custom root finders

Currently only [NewtonRaphson][pardax.NewtonRaphson] is provided as a
root-finding algorithm.

Users can extend the root finders by subclassing
[AbstractRootFinder][pardax.AbstractRootFinder].

## Custom linear solvers

The root finders often use a linear solver as a subroutine.

The linear solvers currently available are:

- [GMRES][pardax.GMRES]
- [CG][pardax.CG]
- [BiCGStab][pardax.BiCGStab]
- [DirectDense][pardax.DirectDense]
- [Spectral][pardax.Spectral]

Users can extend the linear solvers by subclassing
[AbstractLinearSolver][pardax.AbstractLinearSolver].

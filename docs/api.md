# pardax API

`pardax` provides time integration methods for PDEs of the form
$$ \frac{\partial y}{\partial t} = f(t, y). $$

::: pardax.solve_ivp

## Type aliases

- **`RHS`** = `Callable[..., Array]` — Right-hand side function `(t, y, *args) -> dy/dt`.
- **`SplitRHS`** = `Dict[str, RHS]` — Split RHS for IMEX-style schemes, e.g. `{"implicit": f_stiff, "explicit": f_nonstiff}`.

## Time-stepping schemes

::: pardax.ForwardEuler

::: pardax.RK4

::: pardax.BackwardEuler

::: pardax.IMEX

::: pardax.AbstractStepper

## Root finders

::: pardax.NewtonRaphson

::: pardax.AbstractRootFinder

## Linear solvers

::: pardax.GMRES

::: pardax.CG

::: pardax.BiCGStab

::: pardax.DirectDense

::: pardax.Spectral

::: pardax.AbstractLinearSolver
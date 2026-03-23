# API reference

## Solvers

Main entry points for integrating ODE systems.

::: pardax.solve_ivp

::: pardax.integrate

## Time-stepping schemes

Explicit, implicit, and split (IMEX) methods. All built-in steppers
inherit from [AbstractStepper][pardax.AbstractStepper]. Custom steppers
can also be used via the [StepperLike][pardax.StepperLike] protocol
(see [Extending the solver](extending.md)).

### Explicit

::: pardax.ForwardEuler

::: pardax.RK4

### Implicit

::: pardax.BackwardEuler

### Split (IMEX)

::: pardax.IMEX

### Base classes

::: pardax.AbstractStepper

::: pardax.StepperLike

## Root finders

Root finders solve the non-linear or linear system that arises at each
implicit time step. [NewtonRaphson][pardax.NewtonRaphson] handles
general non-linear problems; [LinearRootFinder][pardax.LinearRootFinder]
solves linear systems in a single step.

::: pardax.NewtonRaphson

::: pardax.LinearRootFinder

::: pardax.AbstractRootFinder

## Linearisers

Linearisers construct the Newton system inside
[NewtonRaphson][pardax.NewtonRaphson]. They bundle a linearisation
strategy (autodiff, user-provided JVP, or dense Jacobian) with a
linear solver.

::: pardax.AutoJVP

::: pardax.JVP

::: pardax.Jacobian

::: pardax.AbstractLineariser

## Linear solvers

Solve the linear system $Ax = b$ that arises during root finding.
Iterative solvers accept both dense matrices and matrix-free operators.

::: pardax.DirectDense

::: pardax.GMRES

::: pardax.CG

::: pardax.BiCGStab

::: pardax.SpectralSolver

::: pardax.AbstractLinearSolver

## Linear operators

Linear operators build the implicit system $(I - h L)$ for use with
[LinearRootFinder][pardax.LinearRootFinder]. Each operator returns the
system in the form expected by its paired linear solver.

::: pardax.DenseOperator

::: pardax.MatrixFreeOperator

::: pardax.SpectralOperator

::: pardax.AbstractLinearOperator

## Transforms

Discrete sine transforms for use with Dirichlet boundary conditions.

::: pardax.transform.dst1

::: pardax.transform.idst1
# Solver interface
from .ivp import solve_ivp, integrate

# Time-stepping schemes
from .timesteppers import (
    StepperLike,
    AbstractStepper,
    ForwardEuler,
    RK4,
    BackwardEuler,
)

# Root-finding algorithms
from .rootfinders import (
    AbstractRootFinder,
    NewtonRaphson,
    LinearRootFinder,
    AbstractLineariser,
    Jacobian,
    JVP,
    AutoJVP,
)

# Linear solvers
from .linalg import (
    # Linear solvers
    AbstractLinearSolver,
    DirectDense,
    GMRES,
    CG,
    BiCGStab,
    SpectralSolver,
    # Linear operators
    AbstractLinearOperator,
    DenseOperator,
    MatrixFreeOperator,
    SpectralOperator,
)

__all__ = [
    "solve_ivp",
    "integrate",
    "StepperLike",
    "AbstractStepper",
    "ForwardEuler",
    "RK4",
    "BackwardEuler",
    "AbstractRootFinder",
    "NewtonRaphson",
    "LinearRootFinder",
    "AbstractLineariser",
    "AutoJVP",
    "JVP",
    "Jacobian",
    "AbstractLinearSolver",
    "DirectDense",
    "GMRES",
    "CG",
    "BiCGStab",
    "SpectralSolver",
    "AbstractLinearOperator",
    "DenseOperator",
    "MatrixFreeOperator",
    "SpectralOperator",
]

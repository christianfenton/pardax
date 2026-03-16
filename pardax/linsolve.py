"""Linear solvers used in root-finding and implicit time stepping schemes."""

import abc
from typing import Callable, Union, Optional

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jax_sparse
from jax import Array


class AbstractLinearSolver(eqx.Module):
    """Base class for linear solvers."""

    @abc.abstractmethod
    def __call__(
        self, 
        A: Union[Array, Callable[[Array], Array]],
        b: Array,
        x0: Optional[Array] = None
    ) -> Array:
        """
        Solve the linear system A*x = b.

        Args:
            A: Dense matrix or linear operator with signature x -> A*x
            b: Right-hand side vector
            x0: Initial guess vector

        Returns:
            Solution vector x such that A*x ≈ b
        """
        raise NotImplementedError


class DirectDense(AbstractLinearSolver):
    """
    Direct solver for dense linear systems.

    Dispatches to `jax.numpy.linalg.solve`.
    Only suitable for small systems where the Jacobian is provided explicitly.
    """

    def __call__(
        self, 
        A: Union[Array, Callable[[Array], Array]],
        b: Array,
        x0: Optional[Array] = None
    ) -> Array:
        """
        Solve A*x = b.

        Args:
            A: Dense matrix
            b: Right-hand side vector
            x0: Ignored (kept for interface compatibility). Can be None.

        Returns:
            Solution x

        Raises:
            TypeError: If A is a callable (linear operator) instead of a matrix
        """
        if callable(A):
            raise TypeError(
                "DirectSolve requires a dense matrix, not a linear operator. "
                "Please provide the Jacobian as a dense matrix (jac_fn), "
                "or use an iterative solver (GMRES, CG, BiCGStab)."
            )

        return jnp.linalg.solve(A, b)


class GMRES(AbstractLinearSolver):
    """Generalised Minimal Residual (GMRES).

    Dispatches to `jax.scipy.sparse.linalg.gmres`.
    Suitable for general non-symmetric systems.
    """
    tol: float = eqx.field(static=True)
    maxiter: int = eqx.field(static=True)

    def __init__(self, tol: float = 1e-6, maxiter: int = 100):
        self.tol = tol
        self.maxiter = maxiter

    def __call__(
        self, 
        A: Union[Array, Callable[[Array], Array]],
        b: Array,
        x0: Optional[Array] = None
    ) -> Array:
        """
        Solve A*x = b using GMRES.

        Args:
            A: Dense matrix or linear operator with signature x -> A*x
            b: Right-hand side vector
            x0: Initial guess vector

        Returns:
            Approximate solution x
        """
        solution, info = jax_sparse.gmres(
            A, b, x0=x0, tol=self.tol, maxiter=self.maxiter
        )
        return solution


class CG(AbstractLinearSolver):
    """Conjugate Gradient (CG).

    Dispatches to `jax.scipy.sparse.linalg.cg`.
    Only suitable for symmetric and positive-definite systems.

    Attributes:
        tol: Convergence tolerance for residual norm
        maxiter: Maximum number of iterations
    """
    tol: float = eqx.field(static=True)
    maxiter: int = eqx.field(static=True)

    def __init__(self, tol: float = 1e-6, maxiter: int = 100):
        self.tol = tol
        self.maxiter = maxiter

    def __call__(
        self, 
        A: Union[Array, Callable[[Array], Array]],
        b: Array,
        x0: Optional[Array] = None
    ) -> Array:
        """
        Solve A*x = b.

        Args:
            A: Dense matrix or linear operator with signature x -> A*x
            b: Right-hand side vector
            x0: Initial guess vector

        Returns:
            Approximate solution x
        """
        solution, info = jax_sparse.cg(
            A, b, x0=x0, tol=self.tol, maxiter=self.maxiter
        )
        return solution


class BiCGStab(AbstractLinearSolver):
    """Stabilised Biconjugate Gradient (BiCGStab).

    Dispatches to `jax.scipy.sparse.linalg.bicgstab`.
    Suitable for non-symmetric systems.

    Attributes:
        tol: Convergence tolerance for residual norm
        maxiter: Maximum number of iterations
    """
    tol: float = eqx.field(static=True)
    maxiter: int = eqx.field(static=True)

    def __init__(self, tol: float = 1e-6, maxiter: int = 100):
        self.tol = tol
        self.maxiter = maxiter

    def __call__(
        self, 
        A: Union[Array, Callable[[Array], Array]],
        b: Array,
        x0: Optional[Array] = None
    ) -> Array:
        """
        Solve A*x = b.

        Args:
            A: Dense matrix or linear operator with signature x -> A*x
            b: Right-hand side vector
            x0: Initial guess vector

        Returns:
            Approximate solution x
        """
        solution, info = jax_sparse.bicgstab(
            A, b, x0=x0, tol=self.tol, maxiter=self.maxiter
        )
        return solution
    

def _identity(x):
    return x


class Spectral(AbstractLinearSolver):
    """Spectral linear solver.
    
    Attributes:
        eigvals (Array): Eigenvalues of the linear operator
        forward (Callable): Forward transformation to diagonal basis
        backward (Callable): Backward transformation to original basis
        constraint (Callable): Ensures system satisfies compatibility condition
    """
    eigvals: Array
    forward: Callable[[Array], Array] = eqx.field(static=True)
    backward: Callable[[Array], Array] = eqx.field(static=True)
    constraint: Callable[[Array], Array] = eqx.field(static=True)

    def __init__(
        self,
        eigvals: Array, 
        forward: Callable[[Array], Array], 
        backward: Callable[[Array], Array], 
        constraint: Callable[[Array], Array] = _identity
    ):
        self.eigvals = eigvals
        self.forward = forward
        self.backward = backward
        self.contraint = constraint

    def __call__(
        self, 
        A: Union[Array, Callable[[Array], Array]], 
        b: Array,
        x0: Optional[Array] = None
    ) -> Array:
        """Solve (I - h * A)x = b."""

        if callable(A):
            raise RuntimeError("Got callable A, where an array is required.")

        symbol = 1.0 - A * self.eigvals
        Ainv = jnp.where(jnp.abs(symbol) > 1e-15, 1.0 / symbol, 0.0)
        b_hat = self.forward(self.constraint(b))
        x_hat = Ainv * b_hat
        return jnp.real(self.backward(x_hat))
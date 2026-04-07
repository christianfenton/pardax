"""
Linear solvers and operators used in
root-finding and implicit time stepping schemes.
"""

import abc
from collections.abc import Callable
from typing import Union, Optional

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jax_sparse
from jaxtyping import Array, Float


class AbstractLinearOperator(eqx.Module):
    """Wraps a linear operator and builds the implicit system (I - h * L).

    Subclasses return the system in the form expected by their paired
    ``AbstractLinearSolver``: a dense matrix, a matvec callable, or
    a spectral symbol array.
    """

    @abc.abstractmethod
    def system(
        self, t: Float[Array, ""], h: Float[Array, ""], args: tuple
    ) -> Union[
        Float[Array, "..."], Callable[[Float[Array, " n"]], Float[Array, " n"]]
    ]:
        """Build and return (I - h * L) for the current time step."""
        raise NotImplementedError


class AbstractLinearSolver(eqx.Module):
    """Base class for linear solvers."""

    @abc.abstractmethod
    def __call__(
        self,
        A: Union[
            Float[Array, "..."],
            Callable[[Float[Array, " n"]], Float[Array, " n"]],
        ],
        b: Float[Array, " n"],
        x0: Optional[Float[Array, " n"]] = None,
    ) -> Float[Array, " n"]:
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


class DenseOperator(AbstractLinearOperator):
    """Linear operator that returns a dense system matrix (I - h * L).

    Pair with ``DirectDense``.

    Attributes:
        op_fn: Callable (t, *args) -> L, returning the operator as a dense matrix.
    """

    op_fn: Callable = eqx.field(static=True)

    def __init__(self, op_fn: Callable) -> None:
        self.op_fn = op_fn

    def system(
        self, t: Float[Array, ""], h: Float[Array, ""], args: tuple
    ) -> Float[Array, "n n"]:
        L = self.op_fn(t, *args)
        return jnp.eye(L.shape[0]) - h * L


class MatrixFreeOperator(AbstractLinearOperator):
    """Linear operator that returns a matrix-free matvec for (I - h * L).

    Pair with an iterative solver (``GMRES``, ``CG``, ``BiCGStab``).

    Attributes:
        op_fn: Callable (t, *args) -> Lv, where Lv is a matvec v -> L @ v.
    """

    op_fn: Callable = eqx.field(static=True)

    def __init__(self, op_fn: Callable) -> None:
        self.op_fn = op_fn

    def system(
        self, t: Float[Array, ""], h: Float[Array, ""], args: tuple
    ) -> Callable[[Float[Array, " n"]], Float[Array, " n"]]:
        Lv = self.op_fn(t, *args)
        return lambda v: v - h * Lv(v)


class DirectDense(AbstractLinearSolver):
    """
    Direct solver for dense linear systems.

    Dispatches to `jax.numpy.linalg.solve`.
    Only suitable for small systems where the Jacobian is provided explicitly.
    """

    def __call__(
        self,
        A: Union[
            Float[Array, "n n"],
            Callable[[Float[Array, " n"]], Float[Array, " n"]],
        ],
        b: Float[Array, " n"],
        x0: Optional[Float[Array, " n"]] = None,
    ) -> Float[Array, " n"]:
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
                "DirectSolve requires a dense matrix, got a callable."
                "Please provide A as a dense matrix, "
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
        A: Union[
            Float[Array, "n n"],
            Callable[[Float[Array, " n"]], Float[Array, " n"]],
        ],
        b: Float[Array, " n"],
        x0: Optional[Float[Array, " n"]] = None,
    ) -> Float[Array, " n"]:
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
        A: Union[
            Float[Array, "n n"],
            Callable[[Float[Array, " n"]], Float[Array, " n"]],
        ],
        b: Float[Array, " n"],
        x0: Optional[Float[Array, " n"]] = None,
    ) -> Float[Array, " n"]:
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
        A: Union[
            Float[Array, "n n"],
            Callable[[Float[Array, " n"]], Float[Array, " n"]],
        ],
        b: Float[Array, " n"],
        x0: Optional[Float[Array, " n"]] = None,
    ) -> Float[Array, " n"]:
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


class SpectralOperator(AbstractLinearOperator):
    """Linear operator diagonalised by a known spectral transform.

    Builds and returns the spectral symbol ``1 - h * eigvals``.
    Pair with ``SpectralSolver``.

    Attributes:
        eigvals: Eigenvalues of the linear operator.
    """

    eigvals: Float[Array, " n"]

    def __init__(self, eigvals: Float[Array, " n"]) -> None:
        self.eigvals = eigvals

    def system(
        self, t: Float[Array, ""], h: Float[Array, ""], args: tuple
    ) -> Float[Array, " n"]:
        return 1.0 - h * self.eigvals


def _identity(x):
    return x


class SpectralSolver(AbstractLinearSolver):
    """Spectral linear solver.

    Solves a diagonalised system by transforming to the spectral basis,
    performing a pointwise division by the symbol, and transforming back.

    Pair with ``SpectralOperator``, which passes the precomputed symbol
    array ``1 - h * eigvals`` as ``A``.

    Attributes:
        forward: Forward transformation to diagonal basis
        backward: Inverse transformation from diagonal basis
        constraint: Pre-processing to enforce compatibility (e.g. mean removal)
    """

    forward: Callable[[Array], Array] = eqx.field(static=True)
    backward: Callable[[Array], Array] = eqx.field(static=True)
    constraint: Callable[[Array], Array] = eqx.field(static=True)

    def __init__(
        self,
        forward: Callable[[Array], Array],
        backward: Callable[[Array], Array],
        constraint: Callable[[Array], Array] = _identity,
    ) -> None:
        self.forward = forward
        self.backward = backward
        self.constraint = constraint

    def __call__(
        self,
        A: Union[
            Float[Array, " n"],
            Callable[[Float[Array, " n"]], Float[Array, " n"]],
        ],
        b: Float[Array, " n"],
        x0: Optional[Float[Array, " n"]] = None,
    ) -> Float[Array, " n"]:
        """Solve the system given the spectral symbol.

        Args:
            A: Symbol array ``1 - h * eigvals`` from ``SpectralOperator``.
            b: Right-hand side vector.
            x0: Ignored.

        Returns:
            Solution x.
        """
        if callable(A):
            raise TypeError(
                "SpectralSolver requires a 1d array, not a callable. "
            )
        if jnp.ndim(A) != jnp.ndim(b):
            raise RuntimeError("Expected A to have the same dimensions as b.")

        Ainv = jnp.where(jnp.abs(A) > 1e-15, 1.0 / A, 0.0)
        b_hat = self.forward(self.constraint(b))
        x_hat = Ainv * b_hat
        return jnp.real(self.backward(x_hat))

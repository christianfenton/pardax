import abc
from collections.abc import Callable
from typing import Protocol, Any, runtime_checkable

import equinox as eqx
from jaxtyping import Array, Float


@runtime_checkable
class StepperLike(Protocol):
    """Protocol for any object with a compatible step method."""

    def step(
        self,
        rhs: Any,
        t: Float[Array, ""],
        y: Float[Array, "*state"],
        h: Float[Array, ""],
        args: tuple = (),
    ) -> Float[Array, "*state"]: ...

    """Advance the solution by one time step.

        Args:
            rhs: Right-hand side of the system of equations
            t: Current time (0-dimensional JAX array)
            y: Current solution
            h: Time step size (0-dimensional JAX array)
            args: Additional arguments to pass to fun

        Returns:
            Solution at t + h
    """


class AbstractStepper(eqx.Module):
    """Base class for single-term time-stepping methods."""

    @abc.abstractmethod
    def step(
        self,
        fun: Callable[..., Float[Array, "*state"]],
        t: Float[Array, ""],
        y: Float[Array, "*state"],
        h: Float[Array, ""],
        args: tuple = (),
    ) -> Float[Array, "*state"]:
        """Advance the solution by one time step.

        Args:
            fun: Right-hand side function (t, y, *args) -> dy/dt
            t: Current time (0-dimensional JAX array)
            y: Current solution
            h: Time step size (0-dimensional JAX array)
            args: Additional arguments to pass to fun

        Returns:
            Solution at t + h
        """
        raise NotImplementedError

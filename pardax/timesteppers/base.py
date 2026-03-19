import abc
from typing import Protocol, Callable, Any, runtime_checkable

import equinox as eqx
from jax import Array


@runtime_checkable
class StepperLike(Protocol):
    """Protocol for any object with a compatible step method."""

    def step(
        self,
        fun: Any,
        t: Array,
        y: Array,
        h: Array,
        args: tuple = (),
    ) -> Array: ...
    """Advance the solution by one time step.
    
        Args:
            fun: Right-hand side of the system of equations
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
        fun: Callable[..., Array],
        t: Array,
        y: Array,
        h: Array,
        args: tuple = (),
    ) -> Array:
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
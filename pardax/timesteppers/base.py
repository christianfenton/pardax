import abc
from collections.abc import Callable
from typing import Protocol, Any, runtime_checkable

import equinox as eqx
from jaxtyping import Array, Float


@runtime_checkable
class StepperLike(Protocol):
    """Protocol for any object with a compatible __call__ method."""

    def __call__(
        self,
        fun: Callable,
        t: Float[Array, ""],
        y: Float[Array, "..."],
        step_size: Float[Array, ""],
        params: Any = None,
    ) -> tuple[Float[Array, "..."], "StepperLike"]: ...

    """Advance the solution by one time step.

        Args:
            fun: Right-hand side of the ODE fun(t, y, params) -> dy/dt
            t: Current time (0-dimensional JAX array)
            y: Current solution
            step_size: Time step size (0-dimensional JAX array)
            params: Parameters passed through to fun

        Returns:
            Tuple of the new solution and stepper instance (y, stepper)
    """


class AbstractStepper(eqx.Module):
    """Base class for single-term time-stepping methods."""

    @abc.abstractmethod
    def __call__(
        self,
        fun: Callable,
        t: Float[Array, ""],
        y: Float[Array, "..."],
        step_size: Float[Array, ""],
        params: Any = None,
    ) -> tuple[Float[Array, "..."], "AbstractStepper"]:
        """Advance the solution by one time step.

        Args:
            fun: Right-hand side function fun(t, y, params) -> dy/dt
            t: Current time (0-dimensional JAX array)
            y: Current solution
            step_size: Time step size (0-dimensional JAX array)
            params: Parameters passed through to fun

        Returns:
            Tuple of the new solution and stepper instance (y, stepper)
        """
        raise NotImplementedError

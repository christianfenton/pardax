import abc
from typing import Callable, Dict

import equinox as eqx
from jax import Array

from .roots import AbstractRootFinder, NewtonRaphson


class AbstractStepper(eqx.Module):
    """Base class for time-stepping methods."""

    @abc.abstractmethod
    def step(
        self,
        fun: Callable,
        t: Array,
        y: Array,
        h: Array,
        args: tuple = ()
    ) -> Array:
        """
        Take a single time step.

        Args:
            fun: Right-hand side function
            t: Current time (0-dimensional JAX array)
            y: Current solution
            h: Time step size (0-dimensional JAX array)
            args: Additional arguments to pass to fun

        Returns:
            Solution at t + h
        """
        raise NotImplementedError
    

class ForwardEuler(AbstractStepper):
    """Forward Euler method."""

    def step(
        self,
        fun: Callable,
        t: Array,
        y: Array,
        h: Array,
        args: tuple = ()
    ) -> Array:
        """
        Perform a single Forward Euler step.

        Computes y_next = y_curr + h * f(t_curr, y_curr, *args).

        Args:
            fun: Right-hand side of system dydt = f(t, y, *args).
            t: Current time. Type: 0-dimensional JAX array.
            y: Current solution.
            h: Time step size. Type: 0-dimensional JAX array.
            args: Additional arguments to pass to fun.

        Returns:
            Solution at t + h.
        """
        return y + h * fun(t, y, *args)


class RK4(AbstractStepper):
    """Fourth (4th) order Runge-Kutta method."""

    def step(
        self,
        fun: Callable,
        t: Array,
        y: Array,
        h: Array,
        args: tuple = ()
    ) -> Array:
        """
        Perform a single RK4 step.

        Args:
            fun: Right-hand side of system dy/dt = f(t, y, *args).
            t: Current time.
            y: Current solution.
            h: Time step size.
            args: Additional arguments to pass to fun.

        Returns:
            Solution at t + h.
        """
        k1 = fun(t, y, *args)
        k2 = fun(t + 0.5 * h, y + 0.5 * h * k1, *args)
        k3 = fun(t + 0.5 * h, y + 0.5 * h * k2, *args)
        k4 = fun(t + h, y + h * k3, *args)
        return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    

class BackwardEuler(AbstractStepper):
    """
    Backward Euler time stepper.

    Solves y_next = y_n + h * f(t_next, y_next, *args) at each time step.

    The root finder owns the linearisation strategy, so BackwardEuler
    only defines the residual and initial guess.
    """
    root_finder: AbstractRootFinder

    def __init__(self, root_finder: AbstractRootFinder = NewtonRaphson()):
        self.root_finder = root_finder

    def step(self, fun, t, y, h, args=()):
        def residual(y_next):
            return y_next - y - h * fun(t + h, y_next, *args)
        y0 = y + h * fun(t, y, *args)
        return self.root_finder(residual, y0, fun=fun, t=t + h, h=h, args=args)
    

class IMEX(eqx.Module):
    """
    Split implicit-explicit (IMEX) time-stepping scheme.

    Splits the ODE into stiff (implicit) and non-stiff (explicit) parts:
        dy/dt = f_explicit(t, y) + f_implicit(t, y)

    The scheme advances the solution in two steps:
        1. Explicit: y_star = y_curr + h * f_explicit(t_curr, u_curr)
        2. Implicit: y_next = y_star + h * f_implicit(t_next, y_next)

    Example:
        ```python
        import pardax

        def explicit_term(t, u, ...):
            return ...

        def implicit_term(t, u, ...):
            return ...

        # Instantiate stepper
        stepper = pardax.IMEX(
            implicit=pardax.BackwardEuler(), explicit=pardax.RK4()
        )

        # Define ODE as a dict
        ode = {'implicit': implicit_term, 'explicit': explicit_term}

        # Solve
        t, y = solve_ivp(ode, t_span, y0, stepper, step_size, args)
        ```
    """
    explicit: AbstractStepper
    implicit: AbstractStepper

    def step(
        self,
        fun: Dict[str, Callable],
        t: Array,
        y: Array,
        h: Array,
        args: tuple = ()
    ) -> Array:
        """
        Advance one time step.

        Args:
            fun: Either a dict with keys 'implicit' and 'explicit', or a callable.
                If a dict, fun['explicit'](t, y, *args) gives the non-stiff term
                and fun['implicit'](t, y, *args) gives the stiff term.
                If a callable, it's treated as the implicit term with zero explicit term.
            t: Current time.
            y: Current solution.
            h: Time step size.
            args: Additional arguments to pass to fun.

        Returns:
            Solution at t + h.
        """
        y_star = self.explicit.step(fun['explicit'], t, y, h, args)
        return self.implicit.step(fun['implicit'], t, y_star, h, args)
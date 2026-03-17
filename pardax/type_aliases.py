from typing import Callable, Dict

from jax import Array

#: Right-hand side function: ``(t, y, *args) -> dy/dt``.
RHS = Callable[..., Array]

#: Split right-hand side for IMEX-style schemes,
#: e.g. ``{"implicit": f_stiff, "explicit": f_nonstiff}``.
SplitRHS = Dict[str, RHS]
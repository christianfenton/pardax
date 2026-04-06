from jaxtyping import Array, Float
import jax.numpy as jnp


def dst1(x: Float[Array, " n"]) -> Float[Array, " n"]:
    """Discrete sine transform (type 1)."""
    N = len(x)
    extended = jnp.concatenate(
        [jnp.array([0.0]), x, jnp.array([0.0]), -x[::-1]]
    )
    rfft_result = jnp.fft.rfft(extended)
    X = -rfft_result[1 : N + 1].imag / 2
    return X


def idst1(X: Float[Array, " n"]) -> Float[Array, " n"]:
    """Inverse discrete sine transform (type 1)."""
    N = len(X)
    extended = jnp.concatenate(
        [jnp.array([0.0]), X, jnp.array([0.0]), -X[::-1]]
    )
    rfft_result = jnp.fft.rfft(extended)
    x = -rfft_result[1 : N + 1].imag / (N + 1)
    return x

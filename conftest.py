import equinox as eqx
import jax
import jax.numpy as jnp
import pardax as pdx


def pytest_markdown_docs_globals():
    return {"eqx": eqx, "jax": jax, "jnp": jnp, "pdx": pdx}

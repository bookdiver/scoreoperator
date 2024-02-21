import pytest
import jax
import jax.numpy as jnp
from flax import linen as nn

from ..FNO import SpectralConv1D, FourierLayer1D, FNO1D

# def test_SpectralConv1D():
#     input_dim = 2
#     output_dim = 2
#     n_modes = 4
#     rng_key = jax.random.PRNGKey(42)
#     spectral_conv = SpectralConv1D(
#         input_dim,
#         output_dim,
#         n_modes,
#     )
#     params = spectral_conv.init({"params": rng_key}, jnp.ones((32, input_dim)))
#     out = spectral_conv.apply(params, jnp.ones((32, input_dim)))
#     assert out.shape == (32, output_dim)
#     assert out.dtype == jnp.float32

# def test_FourierLayer1D():
#     input_dim = 2
#     output_dim = 2
#     n_modes = 4
#     activation = nn.relu
#     rng_key = jax.random.PRNGKey(42)
#     fourier_layer = FourierLayer1D(
#         input_dim,
#         output_dim,
#         n_modes,
#         activation,
#     )
#     params = fourier_layer.init({"params": rng_key}, jnp.ones((32, input_dim)))
#     out = fourier_layer.apply(params, jnp.ones((32, input_dim)))
#     assert out.shape == (32, output_dim)
#     assert out.dtype == jnp.float32

def test_FNO1D():
    output_dim = 2
    lifting_dims = [8, 8, 8]
    max_n_modes = [8, 8]
    activation = nn.relu
    rng_key = jax.random.PRNGKey(42)
    fno = FNO1D(
        output_dim,
        lifting_dims,
        max_n_modes,
        activation,
    )
    params = fno.init({"params": rng_key}, jnp.ones((32, 2)))
    out = fno.apply(params, jnp.ones((32, 2)))
    assert out.shape == (32, output_dim)
    assert out.dtype == jnp.float32

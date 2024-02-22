import pytest
import jax
import jax.numpy as jnp
from flax import linen as nn
import matplotlib.pyplot as plt

from ..CTFNO import  TimeEncoding, TMSpectralConv1D, TMFourierLayer1D, TMFNO1D

def test_TimeEncoding():
    encoding_dim = 64
    time_encoding = TimeEncoding(encoding_dim)
    rng_key = jax.random.PRNGKey(42) 
    params = time_encoding.init({"params": rng_key}, jnp.ones((16,)))
    pe = time_encoding.apply(params, jnp.array([0.3]))
    assert pe.shape == (1, encoding_dim)
    ts = jnp.linspace(0, 1, 100)
    pe = time_encoding.apply(params, ts)
    plt.imshow(pe, cmap='RdGy')
    plt.show()
    assert pe.shape == (100, encoding_dim)


def test_TMSpectralConv1D():
    input_dim = 2
    output_dim = 2
    encoding_dim = 32
    n_modes = 4
    rng_key = jax.random.PRNGKey(42)
    spectral_conv = TMSpectralConv1D(
        input_dim,
        output_dim,
        encoding_dim,
        n_modes,
    )
    x = jnp.ones((4, 32, input_dim))
    t_emb = jnp.ones((4, encoding_dim))
    params = spectral_conv.init({"params": rng_key}, x, t_emb)
    out = spectral_conv.apply(params, x, t_emb)
    assert out.shape == (4, 32, output_dim)
    assert out.dtype == jnp.float32

def test_TMFourierLayer1D():
    input_dim = 2
    output_dim = 2
    encoding_dim = 32
    n_modes = 4
    activation = jax.nn.relu
    rng_key = jax.random.PRNGKey(42)
    fourier_layer = TMFourierLayer1D(
        input_dim,
        output_dim,
        encoding_dim,
        n_modes,
        activation,
    )
    x = jnp.ones((4, 32, input_dim))
    t_emb = jnp.ones((4, encoding_dim))
    params = fourier_layer.init({"params": rng_key}, x, t_emb)
    out = fourier_layer.apply(params, x, t_emb)
    assert out.shape == (4, 32, output_dim)
    assert out.dtype == jnp.float32

def test_TMFNO1D():
    output_dim = 2
    lifting_dims = [4, 8, 16]
    max_n_modes = [8, 16]
    encoding_dim = 64
    activation = nn.relu
    rng_key = jax.random.PRNGKey(42)
    tmfno = TMFNO1D(
        output_dim,
        lifting_dims,
        max_n_modes,
        encoding_dim,
        activation,
    )
    x = jnp.ones((1, 32, 2))
    t = jnp.array([0.3])
    params = tmfno.init({"params": rng_key}, x, t)
    out = tmfno.apply(params, x, t)
    assert out.shape == (1, 32, output_dim)
    assert out.dtype == jnp.float32
    x = jnp.ones((4, 64, 2))
    t = jnp.ones((4, ))
    out = tmfno.apply(params, x, t)
    assert out.shape == (4, 64, output_dim)
    assert out.dtype == jnp.float32
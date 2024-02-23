import pytest
import jax
import jax.numpy as jnp
from flax import linen as nn
import matplotlib.pyplot as plt

from ..models.blocks import *

# set jax backend to be cpu
jax.config.update('jax_platform_name', 'cpu')

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

def test_SpectralConv1D():
    input_dim = 2
    output_dim = 2
    n_modes = 4
    rng_key = jax.random.PRNGKey(42)
    spectral_conv = SpectralConv1D(
        input_dim,
        output_dim,
        n_modes,
    )
    x = jnp.ones((4, 32, input_dim))
    print(spectral_conv.tabulate({"params": rng_key}, x))
    params = spectral_conv.init({"params": rng_key}, x)
    out = spectral_conv.apply(params, x)
    assert out.shape == (4, 32, output_dim)
    assert out.dtype == jnp.float32

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
    print(spectral_conv.tabulate({"params": rng_key}, x, t_emb))
    params = spectral_conv.init({"params": rng_key}, x, t_emb)
    out = spectral_conv.apply(params, x, t_emb)
    assert out.shape == (4, 32, output_dim)
    assert out.dtype == jnp.float32

def test_FourierBlock1D():
    input_dim = 2
    output_dim = 2
    n_modes = 4
    activation = nn.relu
    rng_key = jax.random.PRNGKey(42)
    fourier_layer = FourierBlock1D(
        input_dim,
        output_dim,
        n_modes,
        activation,
    )
    print(fourier_layer.tabulate({"params": rng_key}, jnp.ones((32, input_dim))))
    params = fourier_layer.init({"params": rng_key}, jnp.ones((32, input_dim)))
    out = fourier_layer.apply(params, jnp.ones((32, input_dim)))
    assert out.shape == (32, output_dim)
    assert out.dtype == jnp.float32

def test_ResidualFourierBlock1D():
    input_dim = 2
    output_dim = 2
    encoding_dim = 32
    n_modes = 4
    activation = nn.relu
    rng_key = jax.random.PRNGKey(42)
    fourier_layer = ResidualFourierBlock1D(
        input_dim,
        output_dim,
        encoding_dim,
        n_modes,
        activation,
    )
    x = jnp.ones((4, 32, input_dim))
    t_emb = jnp.ones((4, encoding_dim))
    print(fourier_layer.tabulate({"params": rng_key}, x, t_emb))
    params = fourier_layer.init({"params": rng_key}, x, t_emb)
    out = fourier_layer.apply(params, x, t_emb)
    assert out.shape == (4, 32, output_dim)
    assert out.dtype == jnp.float32

def test_TMFourierBlock1D():
    input_dim = 2
    output_dim = 2
    encoding_dim = 32
    n_modes = 4
    activation = jax.nn.relu
    rng_key = jax.random.PRNGKey(42)
    fourier_layer = TMFourierBlock1D(
        input_dim,
        output_dim,
        encoding_dim,
        n_modes,
        activation,
    )
    x = jnp.ones((4, 32, input_dim))
    t_emb = jnp.ones((4, encoding_dim))
    print(fourier_layer.tabulate({"params": rng_key}, x, t_emb))
    params = fourier_layer.init({"params": rng_key}, x, t_emb)
    out = fourier_layer.apply(params, x, t_emb)
    assert out.shape == (4, 32, output_dim)
    assert out.dtype == jnp.float32
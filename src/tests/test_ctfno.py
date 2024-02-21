import pytest
import jax
import jax.numpy as jnp

from ..CTFNO import position_encoding, TMSpectralConv1D, TMFourierLayer1D

def test_position_encoding():
    encoding_dim = 32
    t = 0.3
    pe = position_encoding(t, encoding_dim)
    assert pe.shape == (1, encoding_dim)
    ts = jnp.linspace(0, 1, 10)
    pe = position_encoding(ts, encoding_dim)
    assert pe.shape == (10, encoding_dim)

def test_TMSpectralConv1D():
    input_dim = 2
    output_dim = 2
    encoding_dim = 32
    n_modes = 4
    rng_key = jax.random.PRNGKey(42)
    t_emb = jnp.ones((encoding_dim, ))
    spectral_conv = TMSpectralConv1D(
        input_dim,
        output_dim,
        encoding_dim,
        n_modes,
    )
    params = spectral_conv.init({"params": rng_key}, jnp.ones((32, input_dim)), t_emb)
    out = spectral_conv.apply(params, jnp.ones((32, input_dim)), t_emb)
    assert out.shape == (32, output_dim)
    assert out.dtype == jnp.float32

def test_TMFourierLayer1D():
    input_dim = 2
    output_dim = 2
    encoding_dim = 32
    n_modes = 4
    activation = jax.nn.relu
    rng_key = jax.random.PRNGKey(42)
    t_emb = jnp.ones((encoding_dim, ))
    fourier_layer = TMFourierLayer1D(
        input_dim,
        output_dim,
        encoding_dim,
        n_modes,
        activation,
    )
    params = fourier_layer.init({"params": rng_key}, jnp.ones((32, input_dim)), t_emb)
    out = fourier_layer.apply(params, jnp.ones((32, input_dim)), t_emb)
    assert out.shape == (32, output_dim)
    assert out.dtype == jnp.float32
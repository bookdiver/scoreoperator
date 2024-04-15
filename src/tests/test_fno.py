import pytest
import jax
import jax.numpy as jnp
from flax import linen as nn
import matplotlib.pyplot as plt

from ..models.nerualop.blocks import *
from ..models.nerualop.FNO import *

# set jax backend to be cpu
jax.config.update('jax_platform_name', 'cpu')

def test_TimeEmbedding():
    embedding_dim = 32
    time_encoding = TimeEmbedding(embedding_dim)
    rng_key = jax.random.PRNGKey(42) 
    params = time_encoding.init({"params": rng_key}, jnp.ones((16,)))
    pe = time_encoding.apply(params, jnp.array([0.3]))
    assert pe.shape == (1, embedding_dim)
    # ts = jax.random.uniform(rng_key, (100,))
    ts = jnp.linspace(0, 1, 100)
    pe = time_encoding.apply(params, ts)
    plt.imshow(pe, cmap='RdGy')
    plt.colorbar()
    assert pe.shape == (100, embedding_dim)

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
    params = spectral_conv.init({"params": rng_key}, x)
    out = spectral_conv.apply(params, x)
    assert out.shape == (4, 32, output_dim)
    assert out.dtype == jnp.float32

def test_TimeModulatedSpectralConv1D():
    input_dim = 2
    output_dim = 2
    encoding_dim = 32
    n_modes = 4
    rng_key = jax.random.PRNGKey(42)
    spectral_conv = TimeModulatedSpectralConv1D(
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

def test_ResidualFourierBlock1D():
    input_dim = 2
    output_dim = 2
    encoding_dim = 32
    n_modes = 4
    activation = "relu"
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
    variables = fourier_layer.init(
        {"params": rng_key}, 
        x, 
        t_emb, 
        train=True
    )
    params, batch_stats = variables["params"], variables["batch_stats"]
    out = fourier_layer.apply(
        {"params": params, "batch_stats": batch_stats},
        x, 
        t_emb, 
        train=True, 
        mutable=["batch_stats"],
    )
    assert out[0].shape == (4, 32, output_dim)
    assert out[0].dtype == jnp.float32

def test_TimeModulatedFourierBlock1D():
    input_dim = 2
    output_dim = 2
    encoding_dim = 32
    n_modes = 4
    activation = "relu"
    rng_key = jax.random.PRNGKey(42)
    fourier_layer = TimeModulatedFourierBlock1D(
        input_dim,
        output_dim,
        encoding_dim,
        n_modes,
        activation,
    )
    x = jnp.ones((4, 32, input_dim))
    t_emb = jnp.ones((4, encoding_dim))
    variables = fourier_layer.init(
        {"params": rng_key}, 
        x, 
        t_emb, 
        train=True
    )
    params, batch_stats = variables["params"], variables["batch_stats"]
    out = fourier_layer.apply(
        {"params": params, "batch_stats": batch_stats},
        x, 
        t_emb, 
        train=True, 
        mutable=["batch_stats"],
    )
    assert out[0].shape == (4, 32, output_dim)
    assert out[0].dtype == jnp.float32


def test_TimeDependentFNO1D():
    output_dim = 2
    lifting_dims = [8, 8, 8, 8]
    max_n_modes = [32, 32, 32]
    time_embedding_dim = 64
    activation = "relu"
    rng_key = jax.random.PRNGKey(42)
    tmfno = TimeDependentFNO1D(
        output_dim,
        lifting_dims,
        max_n_modes,
        activation,
        'time_modulated',
        time_embedding_dim,
    )
    x = jnp.ones((2, 64, 2))
    t = jnp.array([0.3, 0.4])
    variables = tmfno.init(
        {"params": rng_key}, 
        x, 
        t, 
        train=True
    )
    params, batch_stats = variables["params"], variables["batch_stats"]
    out = tmfno.apply(
        {"params": params, "batch_stats": batch_stats},
        x, 
        t, 
        train=True,
        mutable=["batch_stats"],
    )
    assert out[0].shape == (2, 64, output_dim)
    assert out[0].dtype == jnp.float32

    x = jnp.ones((4, 128, 2))
    t = jnp.ones((4, ))
    out = tmfno.apply(
        {"params": params, "batch_stats": batch_stats},
        x, 
        t, 
        train=True,
        mutable=["batch_stats"],
    )
    assert out[0].shape == (4, 128, output_dim)
    assert out[0].dtype == jnp.float32

    tmfno = TimeDependentFNO1D(
        output_dim,
        lifting_dims,
        max_n_modes,
        activation,
        'resnet',
        time_embedding_dim,
    )
    x = jnp.ones((2, 64, 2))
    t = jnp.array([0.3, 0.4])
    variables = tmfno.init(
        {"params": rng_key}, 
        x, 
        t, 
        train=True
    )
    params, batch_stats = variables["params"], variables["batch_stats"]
    out = tmfno.apply(
        {"params": params, "batch_stats": batch_stats},
        x, 
        t, 
        train=True,
        mutable=["batch_stats"],
    )
    assert out[0].shape == (2, 64, output_dim)
    assert out[0].dtype == jnp.float32

    x = jnp.ones((4, 128, 2))
    t = jnp.ones((4, ))
    out = tmfno.apply(
        {"params": params, "batch_stats": batch_stats},
        x, 
        t, 
        train=True,
        mutable=["batch_stats"],
    )
    assert out[0].shape == (4, 128, output_dim)
    assert out[0].dtype == jnp.float32
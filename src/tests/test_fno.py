import pytest
import jax
import jax.numpy as jnp
from flax import linen as nn

from ..models.FNO import *

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

def test_TimeDependentFNO1D():
    output_dim = 2
    lifting_dims = [4, 8, 16]
    max_n_modes = [8, 16]
    time_encoding_dim = 64
    activation = nn.relu
    rng_key = jax.random.PRNGKey(42)
    tmfno = TimeDependentFNO1D(
        output_dim,
        lifting_dims,
        max_n_modes,
        activation,
        'time_modulated',
        time_encoding_dim,
    )
    x = jnp.ones((1, 64, 2))
    t = jnp.array([0.3])
    print(tmfno.tabulate({"params": rng_key}, x, t))
    params = tmfno.init({"params": rng_key}, x, t)
    out = tmfno.apply(params, x, t)
    assert out.shape == (1, 64, output_dim)
    assert out.dtype == jnp.float32
    x = jnp.ones((4, 128, 2))
    t = jnp.ones((4, ))
    out = tmfno.apply(params, x, t)
    assert out.shape == (4, 128, output_dim)
    assert out.dtype == jnp.float32

    tmfno = TimeDependentFNO1D(
        output_dim,
        lifting_dims,
        max_n_modes,
        activation,
        'resnet',
        time_encoding_dim,
    )
    x = jnp.ones((1, 64, 2))
    t = jnp.array([0.3])
    print(tmfno.tabulate({"params": rng_key}, x, t))
    params = tmfno.init({"params": rng_key}, x, t)
    out = tmfno.apply(params, x, t)
    assert out.shape == (1, 64, output_dim)
    assert out.dtype == jnp.float32
    x = jnp.ones((4, 128, 2))
    t = jnp.ones((4, ))
    out = tmfno.apply(params, x, t)
    assert out.shape == (4, 128, output_dim)
    assert out.dtype == jnp.float32
import pytest
import jax
import jax.numpy as jnp
from flax import linen as nn
from bridgeop.neuralop.blocks import (
    SpectralConv,
    FMSpectralConv,
    TimeConv,
    CTUNOBlock,
    TimeEmbedding,
    get_activation_fn,
)

@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)

def test_get_activation_fn():
    assert get_activation_fn('relu') == nn.relu
    assert get_activation_fn('tanh') == nn.tanh
    assert get_activation_fn('silu') == nn.silu
    assert get_activation_fn('gelu') == nn.gelu
    assert get_activation_fn('leaky_relu') == nn.leaky_relu
    assert get_activation_fn('elu') == nn.elu
    
    with pytest.raises(ValueError):
        get_activation_fn('invalid_activation')

def test_spectral_conv(rng):
    batch_size, in_grid_sz, in_co_dim = 2, 32, 4
    out_co_dim, n_modes = 8, 16
    
    x = jax.random.normal(rng, (batch_size, in_grid_sz, in_co_dim))
    
    model = SpectralConv(in_co_dim, out_co_dim, n_modes)
    params = model.init(rng, x)
    
    output = model.apply(params, x)
    assert output.shape == (batch_size, in_grid_sz, out_co_dim)

def test_fm_spectral_conv(rng):
    batch_size, in_grid_sz, in_co_dim, t_emb_dim = 2, 32, 4, 10
    out_co_dim, n_modes = 8, 16
    
    x = jax.random.normal(rng, (batch_size, in_grid_sz, in_co_dim))
    phi_t = jax.random.normal(rng, (batch_size, t_emb_dim))
    
    model = FMSpectralConv(in_co_dim, out_co_dim, t_emb_dim, n_modes)
    params = model.init(rng, x, phi_t)
    
    output = model.apply(params, x, phi_t)
    assert output.shape == (batch_size, in_grid_sz, out_co_dim)

def test_time_conv(rng):
    batch_size, in_grid_sz, in_co_dim, t_emb_dim = 2, 32, 4, 10
    out_co_dim = 8
    
    x = jax.random.normal(rng, (batch_size, in_grid_sz, in_co_dim))
    psi_t = jax.random.normal(rng, (batch_size, t_emb_dim))
    
    model = TimeConv(out_co_dim)
    params = model.init(rng, x, psi_t)
    
    output = model.apply(params, x, psi_t)
    assert output.shape == (batch_size, in_grid_sz, out_co_dim)

def test_ctuno_block(rng):
    batch_size, in_grid_sz, in_co_dim, t_emb_dim = 2, 32, 4, 10
    out_co_dim, n_modes = 8, 16
    
    x = jax.random.normal(rng, (batch_size, in_grid_sz, in_co_dim))
    t_emb = jax.random.normal(rng, (batch_size, t_emb_dim))
    
    model = CTUNOBlock(in_co_dim, out_co_dim, t_emb_dim, n_modes)
    variables = model.init(rng, x, t_emb)
    
    # Separate params and batch_stats
    params, batch_stats = variables['params'], variables['batch_stats']
    
    # Use apply_fn with mutable=['batch_stats'] for training
    output, updated_batch_stats = model.apply(
        {'params': params, 'batch_stats': batch_stats},
        x, t_emb,
        mutable=['batch_stats'],
        rngs={'dropout': rng}
    )
    assert output.shape == (batch_size, in_grid_sz, out_co_dim)
    
    # Use apply_fn with train=False for inference
    inference_output = model.apply(
        {'params': params, 'batch_stats': batch_stats},
        x, t_emb,
        train=False,
        rngs={'dropout': rng}
    )
    assert inference_output.shape == (batch_size, in_grid_sz, out_co_dim)

def test_time_embedding(rng):
    batch_size, t_emb_dim = 2, 10
    
    t = jax.random.uniform(rng, (batch_size,))
    
    model = TimeEmbedding(t_emb_dim)
    params = model.init(rng, t)
    
    output = model.apply(params, t)
    assert output.shape == (batch_size, t_emb_dim)

# Additional tests for edge cases and specific behaviors
def test_spectral_conv_different_output_size(rng):
    batch_size, in_grid_sz, in_co_dim = 2, 32, 4
    out_co_dim, n_modes, out_grid_sz = 8, 16, 64
    
    x = jax.random.normal(rng, (batch_size, in_grid_sz, in_co_dim))
    
    model = SpectralConv(in_co_dim, out_co_dim, n_modes, out_grid_sz=out_grid_sz)
    params = model.init(rng, x)
    
    output = model.apply(params, x)
    assert output.shape == (batch_size, out_grid_sz, out_co_dim)

def test_ctuno_block_no_freq_mod(rng):
    batch_size, in_grid_sz, in_co_dim, t_emb_dim = 2, 32, 4, 10
    out_co_dim, n_modes = 8, 16
    
    x = jax.random.normal(rng, (batch_size, in_grid_sz, in_co_dim))
    t_emb = jax.random.normal(rng, (batch_size, t_emb_dim))
    
    model = CTUNOBlock(in_co_dim, out_co_dim, t_emb_dim, n_modes, use_freq_mod=False)
    variables = model.init(rng, x, t_emb)
    
    # Separate params and batch_stats
    params, batch_stats = variables['params'], variables['batch_stats']
    
    # Use apply_fn with mutable=['batch_stats'] for training
    output, updated_batch_stats = model.apply(
        {'params': params, 'batch_stats': batch_stats},
        x, t_emb,
        mutable=['batch_stats'],
        rngs={'dropout': rng}
    )
    assert output.shape == (batch_size, in_grid_sz, out_co_dim)
    
    # Use apply_fn with train=False for inference
    inference_output = model.apply(
        {'params': params, 'batch_stats': batch_stats},
        x, t_emb,
        train=False,
        rngs={'dropout': rng}
    )
    assert inference_output.shape == (batch_size, in_grid_sz, out_co_dim)

def test_ctuno_block_instance_norm(rng):
    batch_size, in_grid_sz, in_co_dim, t_emb_dim = 2, 32, 4, 10
    out_co_dim, n_modes = 8, 16
    
    x = jax.random.normal(rng, (batch_size, in_grid_sz, in_co_dim))
    t_emb = jax.random.normal(rng, (batch_size, t_emb_dim))
    
    model = CTUNOBlock(in_co_dim, out_co_dim, t_emb_dim, n_modes, norm="instance")
    params = model.init(rng, x, t_emb)
    
    output = model.apply(params, x, t_emb)
    assert output.shape == (batch_size, in_grid_sz, out_co_dim)

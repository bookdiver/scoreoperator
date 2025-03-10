import pytest
import jax
import jax.numpy as jnp
from flax import linen as nn
from bridgeop.neuralop.blocks import (
    SpectralConv1D,
    SpectralConv2D,
    FMSpectralConv1D,
    FMSpectralConv2D,
    TimeConv1D,
    TimeConv2D,
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
    batch_size, in_grid_szs, in_co_dim = 2, (32, ), 4
    out_co_dim, n_modes = 8, (16, )
    
    x = jax.random.normal(rng, (batch_size, *in_grid_szs, in_co_dim))
    
    model = SpectralConv1D(in_co_dim, out_co_dim, n_modes)
    params = model.init(rng, x)
    
    output = model.apply(params, x)
    assert output.shape == (batch_size, *in_grid_szs, out_co_dim)
    
    batch_size, in_grid_szs, in_co_dim = 2, (32, 32), 3
    out_co_dim, n_modes = 6, (12, 12)
    
    x = jax.random.normal(rng, (batch_size, *in_grid_szs, in_co_dim))
    
    model = SpectralConv2D(in_co_dim, out_co_dim, n_modes)
    params = model.init(rng, x)
    
    output = model.apply(params, x)
    assert output.shape == (batch_size, *in_grid_szs, out_co_dim)

def test_fm_spectral_conv(rng):
    batch_size, in_grid_szs, in_co_dim, t_emb_dim = 2, (32, ), 4, 10
    out_co_dim, n_modes = 8, (16, )
    
    x = jax.random.normal(rng, (batch_size, *in_grid_szs, in_co_dim))
    phi_t = jax.random.normal(rng, (batch_size, t_emb_dim))
    
    model = FMSpectralConv1D(in_co_dim, out_co_dim, t_emb_dim, n_modes)
    params = model.init(rng, x, phi_t)
    
    output = model.apply(params, x, phi_t)
    assert output.shape == (batch_size, *in_grid_szs, out_co_dim)
    
    batch_size, in_grid_szs, in_co_dim = 2, (32, 32), 3
    out_co_dim, n_modes = 6, (12, 12)
    
    x = jax.random.normal(rng, (batch_size, *in_grid_szs, in_co_dim))
    phi_t = jax.random.normal(rng, (batch_size, t_emb_dim))
    
    model = FMSpectralConv2D(in_co_dim, out_co_dim, t_emb_dim, n_modes)
    params = model.init(rng, x, phi_t)
    
    output = model.apply(params, x, phi_t)
    assert output.shape == (batch_size, *in_grid_szs, out_co_dim)

def test_time_conv(rng):
    batch_size, in_grid_szs, in_co_dim, t_emb_dim = 2, (32, ), 4, 10
    out_co_dim = 8
    
    x = jax.random.normal(rng, (batch_size, *in_grid_szs, in_co_dim))
    psi_t = jax.random.normal(rng, (batch_size, t_emb_dim))
    
    model = TimeConv1D(out_co_dim)
    params = model.init(rng, x, psi_t)
    
    output = model.apply(params, x, psi_t)
    assert output.shape == (batch_size, *in_grid_szs, out_co_dim)
    
    batch_size, in_grid_szs, in_co_dim = 2, (32, 32), 3
    out_co_dim, t_emb_dim = 6, 10
    
    x = jax.random.normal(rng, (batch_size, *in_grid_szs, in_co_dim))
    psi_t = jax.random.normal(rng, (batch_size, t_emb_dim))
    
    model = TimeConv2D(out_co_dim)
    params = model.init(rng, x, psi_t)
    
    output = model.apply(params, x, psi_t)
    assert output.shape == (batch_size, *in_grid_szs, out_co_dim)

def test_ctuno_block(rng):
    do_dim = 1
    batch_size, in_grid_szs, in_co_dim = 2, (32, ), 4
    t_emb_dim = 32
    out_co_dim, n_modes = 8, (16, )
    out_grid_scaling = 0.5
    
    x = jax.random.normal(rng, (batch_size, *in_grid_szs, in_co_dim))
    t_emb = jax.random.normal(rng, (batch_size, t_emb_dim))
    
    model = CTUNOBlock(do_dim, in_co_dim, out_co_dim, t_emb_dim, n_modes, out_grid_scaling)
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
    assert output.shape == (batch_size, *tuple(int(out_grid_scaling * in_grid_sz) for in_grid_sz in in_grid_szs), out_co_dim)
    
    # Use apply_fn with train=False for inference
    in_grid_szs = (64, )
    x = jax.random.normal(rng, (batch_size, *in_grid_szs, in_co_dim))
    inference_output = model.apply(
        {'params': params, 'batch_stats': batch_stats},
        x, t_emb,
        train=False,
        rngs={'dropout': rng}
    )
    assert inference_output.shape == (batch_size, *tuple(int(out_grid_scaling * in_grid_sz) for in_grid_sz in in_grid_szs), out_co_dim)
    
    # do_dim = 2
    # batch_size, in_grid_szs, in_co_dim, t_emb_dim = 2, (32, 32), 4, 10
    # out_co_dim, n_modes = 4, (12, 12)
    
    # x = jax.random.normal(rng, (batch_size, *in_grid_szs, in_co_dim))
    # t_emb = jax.random.normal(rng, (batch_size, t_emb_dim))
    
    # model = CTUNOBlock(do_dim, in_co_dim, out_co_dim, t_emb_dim, n_modes)
    # variables = model.init(rng, x, t_emb)
    
    # # Separate params and batch_stats
    # params, batch_stats = variables['params'], variables['batch_stats']
    
    # # Use apply_fn with mutable=['batch_stats'] for training
    # output, updated_batch_stats = model.apply(
    #     {'params': params, 'batch_stats': batch_stats},
    #     x, t_emb,
    #     mutable=['batch_stats'],
    #     rngs={'dropout': rng}
    # )
    # assert output.shape == (batch_size, *in_grid_szs, out_co_dim)
    
    # # Use apply_fn with train=False for inference
    # inference_output = model.apply(
    #     {'params': params, 'batch_stats': batch_stats},
    #     x, t_emb,
    #     train=False,
    #     rngs={'dropout': rng}
    # )
    # assert inference_output.shape == (batch_size, *in_grid_szs, out_co_dim)

def test_time_embedding(rng):
    batch_size, t_emb_dim = 2, 10
    
    t = jax.random.uniform(rng, (batch_size,))
    
    model = TimeEmbedding(t_emb_dim)
    params = model.init(rng, t)
    
    output = model.apply(params, t)
    assert output.shape == (batch_size, t_emb_dim)

# Additional tests for edge cases and specific behaviors
def test_spectral_conv_different_output_size(rng):
    batch_size, in_grid_szs, in_co_dim = 2, (32, ), 4
    out_co_dim, n_modes, out_grid_szs = 8, (16, ), (64, )
    
    x = jax.random.normal(rng, (batch_size, *in_grid_szs, in_co_dim))
    
    model = SpectralConv1D(in_co_dim, out_co_dim, n_modes, out_grid_szs=out_grid_szs)
    params = model.init(rng, x)
    
    output = model.apply(params, x)
    assert output.shape == (batch_size, *out_grid_szs, out_co_dim)
    
    batch_size, in_grid_szs, in_co_dim = 2, (32, 32), 3
    out_co_dim, n_modes, out_grid_szs = 6, (12, 12), (24, 24)
    
    x = jax.random.normal(rng, (batch_size, *in_grid_szs, in_co_dim))
    
    model = SpectralConv2D(in_co_dim, out_co_dim, n_modes, out_grid_szs=out_grid_szs)
    params = model.init(rng, x)
    
    output = model.apply(params, x)
    assert output.shape == (batch_size, *out_grid_szs, out_co_dim)
    
def test_ctuno_block_no_freq_mod(rng):
    batch_size, in_grid_szs, in_co_dim, t_emb_dim = 2, (32, ), 4, 10
    out_co_dim, n_modes = 8, (16, )
    
    x = jax.random.normal(rng, (batch_size, *in_grid_szs, in_co_dim))
    t_emb = jax.random.normal(rng, (batch_size, t_emb_dim))
    
    model = CTUNOBlock1D(in_co_dim, out_co_dim, t_emb_dim, n_modes, use_freq_mod=False)
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
    
    model = CTUNOBlock1D(in_co_dim, out_co_dim, t_emb_dim, n_modes, norm="instance")
    params = model.init(rng, x, t_emb)
    
    output = model.apply(params, x, t_emb)
    assert output.shape == (batch_size, in_grid_sz, out_co_dim)

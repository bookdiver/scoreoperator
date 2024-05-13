import jax
import jax.numpy as jnp

from ..models.neuralop.blocks import *
from ..models.neuralop.uno import *


# set jax backend to be cpu
jax.config.update('jax_platform_name', 'cpu')

def count_params(params):
    return sum(x.size for x in jax.tree_leaves(params))

def test_TimeEmbedding():
    t_emb_dim = 32
    time_embedding = TimeEmbedding(t_emb_dim)
    rng_key = jax.random.PRNGKey(42) 
    out, params = time_embedding.init_with_output({"params": rng_key}, jnp.ones((16,)))
    assert out.shape == (16, t_emb_dim)

def test_SpectralConv1D():
    in_co_dim = 2
    out_co_dim = 2
    n_modes = 4
    rng_key = jax.random.PRNGKey(42)
    spectral_conv = SpectralConv1D(
        in_co_dim,
        out_co_dim,
        n_modes,
    )
    x = jnp.ones((4, 32, in_co_dim))
    out, params = spectral_conv.init_with_output({"params": rng_key}, x)
    assert out.shape == (4, 32, out_co_dim)
    assert out.dtype == jnp.float32

def test_SpectralConv2D():
    in_co_dim = 1
    out_co_dim = 2
    n_modes = 16
    rng_key = jax.random.PRNGKey(42)
    spectral_conv = SpectralConv2D(
        in_co_dim,
        out_co_dim,
        n_modes,
    )
    x = jnp.ones((4, 32, 32, in_co_dim))
    out, params = spectral_conv.init_with_output({"params": rng_key}, x)
    assert out.shape == (4, 32, 32, out_co_dim)
    assert out.dtype == jnp.float32
    
def test_SpectralFreqTimeConv1D():
    in_co_dim = 2
    out_co_dim = 2
    n_modes = 4
    t_emb_dim = 32
    rng_key = jax.random.PRNGKey(42)
    spectral_conv = SpectralFreqTimeConv1D(
        in_co_dim,
        out_co_dim,
        t_emb_dim,
        n_modes,
    )
    x = jnp.ones((4, 32, in_co_dim))
    t_emb = jnp.ones((4, t_emb_dim))
    out, params = spectral_conv.init_with_output({"params": rng_key}, x, t_emb)
    assert out.shape == (4, 32, out_co_dim)
    assert out.dtype == jnp.float32

def test_SpectralFreqTimeConv2D():
    in_co_dim = 1
    out_co_dim = 2
    n_modes = 8
    t_emb_dim = 32
    rng_key = jax.random.PRNGKey(42)
    spectral_conv = SpectralFreqTimeConv2D(
        in_co_dim,
        out_co_dim,
        t_emb_dim,
        n_modes,
    )
    x = jnp.ones((4, 32, 32, in_co_dim))
    t_emb = jnp.ones((4, t_emb_dim))
    out, params = spectral_conv.init_with_output({"params": rng_key}, x, t_emb)
    assert out.shape == (4, 32, 32, out_co_dim)
    assert out.dtype == jnp.float32

def test_CTUNOBlock1D():
    in_co_dim = 2
    out_co_dim = 2
    t_emb_dim = 32
    n_modes = 4
    out_grid_sz = 64
    fft_norm = "forward"
    norm = "batch"
    act = "relu"
    rng_key = jax.random.PRNGKey(42)
    tmfblock = CTUNOBlock1D(
        in_co_dim,
        out_co_dim,
        t_emb_dim,
        n_modes,
        out_grid_sz,
        fft_norm,
        norm,
        act,
    )
    x = jnp.ones((4, 32, in_co_dim))
    t_emb = jnp.ones((4, t_emb_dim))
    out, params = tmfblock.init_with_output({"params": rng_key}, x, t_emb)
    assert out.shape == (4, out_grid_sz, out_co_dim)
    assert out.dtype == jnp.float32


def test_CTUNOBlock2D():
    in_co_dim = 2
    out_co_dim = 2
    t_emb_dim = 32
    n_modes = 8
    out_grid_sz = 64
    fft_norm = "forward"
    norm = "batch"
    act = "relu"
    rng_key = jax.random.PRNGKey(42)
    tmfblock = CTUNOBlock2D(
        in_co_dim,
        out_co_dim,
        t_emb_dim,
        n_modes,
        out_grid_sz,
        fft_norm,
        norm,
        act,
    )
    x = jnp.ones((4, 32, 32, in_co_dim))
    t_emb = jnp.ones((4, t_emb_dim))
    out, params = tmfblock.init_with_output({"params": rng_key}, x, t_emb)
    assert out.shape == (4, out_grid_sz, out_grid_sz, out_co_dim)
    assert out.dtype == jnp.float32

def test_CTUNO1D():
    in_co_dim = 2
    out_co_dim = 2
    lifting_dim = 16
    n_modes_per_layer = [8, 4, 2]
    co_dims_fmults = [1, 2, 4]
    norm = "batch"
    act = "relu"
    rng_key = jax.random.PRNGKey(42)
    uno = CTUNO1D(
        out_co_dim,
        lifting_dim,
        co_dims_fmults,
        n_modes_per_layer,
        norm,
        act,
    )
    x = jnp.ones((4, 32, in_co_dim))
    t_emb = jnp.ones((4, ))
    train = True
    out, params = uno.init_with_output({"params": rng_key}, x, t_emb, train)
    assert out.shape == (4, 32, out_co_dim)
    assert out.dtype == jnp.float32


def test_CTUNO2D():
    in_co_dim = 2
    out_co_dim = 2
    lifting_dim = 16
    n_modes_per_layer = [8, 4, 2]
    co_dims_fmults = [1, 2, 4]
    norm = "batch"
    act = "relu"
    rng_key = jax.random.PRNGKey(42)
    uno = CTUNO2D(
        out_co_dim,
        lifting_dim,
        co_dims_fmults,
        n_modes_per_layer,
        norm,
        act,
    )
    x = jnp.ones((4, 32, 32, in_co_dim))
    t_emb = jnp.ones((4, ))
    train = True
    out, params = uno.init_with_output({"params": rng_key}, x, t_emb, train)
    print(count_params(params))
    assert out.shape == (4, 32, 32, out_co_dim)
    assert out.dtype == jnp.float32
    x = jnp.ones((4, 64, 64, in_co_dim))
    out = uno.apply(params, x, t_emb, train=False)
    assert out.shape == (4, 64, 64, out_co_dim)
    assert out.dtype == jnp.float32
    
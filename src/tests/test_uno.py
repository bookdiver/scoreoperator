import pytest
import jax
import jax.numpy as jnp
from flax import linen as nn
from bridgeop.neuralop.uno import CTUNO

@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)

def test_ctuno_initialization(rng):
    in_co_dim = 3
    out_co_dim = 3
    lifting_dim = 32
    co_dims_fmults = (2, 4, 8)
    n_modes_per_layer = (16, 12, 8)

    model = CTUNO(
        in_co_dim=in_co_dim,
        out_co_dim=out_co_dim,
        lifting_dim=lifting_dim,
        co_dims_fmults=co_dims_fmults,
        n_modes_per_layer=n_modes_per_layer
    )

    batch_sz, in_grid_sz = 16, 64
    x = jax.random.normal(rng, (batch_sz, in_grid_sz * in_co_dim))
    t = jax.random.uniform(rng, (batch_sz,))

    variables = model.init(rng, x, t)
    assert 'params' in variables
    if model.norm.lower() == "batch":
        assert 'batch_stats' in variables

def test_ctuno_forward_pass(rng):
    in_co_dim = 3
    out_co_dim = 3
    lifting_dim = 32
    co_dims_fmults = (2, 4, 4)
    n_modes_per_layer = (10, 8, 4)

    model = CTUNO(
        in_co_dim=in_co_dim,
        out_co_dim=out_co_dim,
        lifting_dim=lifting_dim,
        co_dims_fmults=co_dims_fmults,
        n_modes_per_layer=n_modes_per_layer
    )

    batch_sz, in_grid_sz = 16, 64
    x = jax.random.normal(rng, (batch_sz, in_grid_sz * in_co_dim))
    t = jax.random.uniform(rng, (batch_sz,))

    variables = model.init(rng, x, t)

    # Training mode
    output, updated_vars = model.apply(variables, x, t, mutable=['batch_stats'])
    assert output.shape == (batch_sz, in_grid_sz * out_co_dim)

    # Inference mode
    batch_sz, in_grid_sz = 16, 32
    x = jax.random.normal(rng, (batch_sz, in_grid_sz * in_co_dim))
    t = jax.random.uniform(rng, (batch_sz,))
    inference_output = model.apply(variables, x, t, train=False)
    assert inference_output.shape == (batch_sz, in_grid_sz * out_co_dim)

def test_ctuno_different_configurations(rng):
    configurations = [
        {
            'in_co_dim': 3,
            'out_co_dim': 3,
            'lifting_dim': 16,
            'co_dims_fmults': (2, 4),
            'n_modes_per_layer': (8, 6),
            'norm': 'instance',
            'act': 'gelu',
            'use_freq_mod': False
        },
        {
            'in_co_dim': 3,
            'out_co_dim': 3,
            'lifting_dim': 64,
            'co_dims_fmults': (2, 4, 8, 8),
            'n_modes_per_layer': (10, 8, 6, 6),
            'norm': 'batch',
            'act': 'silu',
            'use_freq_mod': True
        }
    ]

    for config in configurations:
        model = CTUNO(**config)

        batch_sz, in_grid_sz, in_co_dim = 2, 64, 3
        x = jax.random.normal(rng, (batch_sz, in_grid_sz * in_co_dim))
        t = jax.random.uniform(rng, (batch_sz,))

        variables = model.init(rng, x, t)

        # Training mode
        output, updated_vars = model.apply(variables, x, t, mutable=['batch_stats'])
        assert output.shape == (batch_sz, in_grid_sz * config['out_co_dim'])

        # Inference mode
        inference_output = model.apply(variables, x, t, train=False)
        assert inference_output.shape == (batch_sz, in_grid_sz * config['out_co_dim'])

def test_ctuno_gradient(rng):
    in_co_dim = 3
    out_co_dim = 3
    lifting_dim = 32
    co_dims_fmults = (2, 4)
    n_modes_per_layer = (10, 8)
    norm = "batch"

    model = CTUNO(
        in_co_dim=in_co_dim,
        out_co_dim=out_co_dim,
        lifting_dim=lifting_dim,
        co_dims_fmults=co_dims_fmults,
        n_modes_per_layer=n_modes_per_layer,
        norm=norm
    )

    batch_sz, in_grid_sz = 2, 64
    x = jax.random.normal(rng, (batch_sz, in_grid_sz * in_co_dim))
    t = jax.random.uniform(rng, (batch_sz,))

    variables = model.init(rng, x, t)

    def loss_fn(params):
        output = model.apply({'params': params, 'batch_stats': variables['batch_stats']}, x, t, train=False)
        return jnp.mean(output**2)

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(variables['params'])

    assert jax.tree_util.tree_structure(grads) == jax.tree_util.tree_structure(variables['params'])
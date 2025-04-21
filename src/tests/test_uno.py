import pytest
import jax
import jax.random as jr

from flax import linen as nn
from bridgeop.neuralop.uno import CTUNO1D, CTUNO2D

@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)

def test_ctuno1d_initialization(rng):
    d_u = 3
    d_v = 3
    c = 16
    d_ls = (16, 32, 64)
    modes_ls = (8, 6)
    

    model = CTUNO1D(
        d_u, d_v, c, d_ls, modes_ls
    )

    b, L = 16, 32
    x = jr.normal(rng, (b, L, d_u))
    t = jr.uniform(rng, (b,))
    
    tabulate_fn = nn.tabulate(
        model, jax.random.key(0), compute_flops=True, compute_vjp_flops=True
    )
    print(tabulate_fn(x, t))
    
def test_ctuno2d_initialization(rng):
    d_u = 3
    d_v = 3
    c = 16
    d_ls = (16, 32, 64)
    modes_ls = (8, 6)
    
    model = CTUNO2D(
        d_u, d_v, c, d_ls, modes_ls
    )
    b, H, W = 16, 32, 32
    x = jr.normal(rng, (b, H, W, d_u))
    t = jr.uniform(rng, (b,))
    
    tabulate_fn = nn.tabulate(
        model, jax.random.key(0), compute_flops=True, compute_vjp_flops=True
    )
    print(tabulate_fn(x, t))
    

def test_ctuno1d_forward_pass(rng):
    d_u = 3
    d_v = 2
    c = 16
    d_ls = (16, 32, 64)
    modes_ls = (8, 6)
    

    model = CTUNO1D(
        d_u, d_v, c, d_ls, modes_ls
    )

    b, L = 16, 32
    x = jr.normal(rng, (b, L, d_u))
    t = jr.uniform(rng, (b,))
    
    variables = model.init(rng, x, t)
    out = model.apply(variables, x, t, train=True)
    assert out.shape == (b, L, d_v)

    # Inference mode
    x = jr.normal(rng, (b, L*2, d_u))
    t = jr.uniform(rng, (b, ))
    output = model.apply(variables, x, t, train=False)
    assert output.shape == (b, L*2, d_v)
    
def test_ctuno2d_forward_pass(rng):
    d_u = 3
    d_v = 2
    c = 16
    d_ls = (16, 32, 64)
    modes_ls = (8, 6)
    

    model = CTUNO2D(
        d_u, d_v, c, d_ls, modes_ls
    )

    b, H, W = 16, 32, 32
    x = jr.normal(rng, (b, H, W, d_u))
    t = jr.uniform(rng, (b,))
    
    variables = model.init(rng, x, t)
    out = model.apply(variables, x, t, train=True)
    assert out.shape == (b, H, W, d_v)

    # Inference mode
    x = jr.normal(rng, (b, H*2, W*2, d_u))
    t = jr.uniform(rng, (b, ))
    output = model.apply(variables, x, t, train=False)
    assert output.shape == (b, H*2, W*2, d_v)
import pytest
import jax
import jax.numpy as jnp
from bridgeop.diffusion.bridge import DiffusionBridge

@pytest.fixture
def diffusion_bridge():
    seed = 42
    sde_name = "brownian"
    sde_kwargs = {"W_shape": (2,), "T": 1.0}
    dt = 0.01
    return DiffusionBridge(seed, sde_name, sde_kwargs, dt)

def test_initialization(diffusion_bridge):
    assert isinstance(diffusion_bridge, DiffusionBridge)
    assert diffusion_bridge.dt == 0.01
    assert isinstance(diffusion_bridge.rng_key, jax.Array)

def test_get_bs(diffusion_bridge):
    xs = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    ts = jnp.array([0.0, 0.01, 0.02])
    bs = diffusion_bridge.get_bs(xs, ts)
    assert bs.shape == (2, 2)

def test_solve_forward_sde(diffusion_bridge):
    rng_key = jax.random.PRNGKey(0)
    x0 = jnp.array([0.0, 0.0])
    n_batches = 100
    sol = diffusion_bridge.solve_forward_sde(rng_key, x0, n_batches)
    assert sol.xs.shape == (n_batches, 100 - 1, 2)
    assert sol.ts.shape == (100 - 1, )
    assert sol.bs.shape == (n_batches, 100 - 1, 2)

def test_dsm_loss(diffusion_bridge):
    rng_key = jax.random.PRNGKey(0)
    x0 = jnp.array([0.0, 0.0])
    n_batches = 64
    sol = diffusion_bridge.solve_forward_sde(rng_key, x0, n_batches)
    xs, bs = sol.xs, sol.bs
    loss = diffusion_bridge.dsm_loss(xs, bs)
    assert isinstance(loss, jnp.ndarray)
    assert loss.shape == ()

from jax import random
import jax.numpy as jnp

from ..models.diffusion.diffuser import Diffuser1D, Diffuser2D
from ..models.diffusion.sde import SDE

def test_diffuser1d_solve_sde():
    seed = 0
    dim = 2
    sde = SDE(name="brownian", sigma=1.0)
    dt = 1e-2
    diffuser = Diffuser1D(seed, sde, dt)

    rng_key = random.PRNGKey(seed)
    x0 = jnp.array([0.0, 0.0])
    xs, ts, grads, _ = diffuser.solve_sde(rng_key, x0)

    # Assert the shapes of the outputs
    assert xs.shape == (100, 2)
    assert ts.shape == (100,)
    assert grads.shape == (100, 2)

def test_diffuser1d_solve_reverse_bridge_sde():
    seed = 0
    sde = SDE(name="brownian", sigma=1.0)
    dt = 1e-2
    diffusion = Diffuser1D(seed, sde, dt)

    rng_key = random.PRNGKey(seed)
    x0 = jnp.array([0.0, 0.0])
    score_fn = lambda t, x: x / (1.0 - t)
    xs, ts = diffusion.solve_reverse_bridge_sde(rng_key, x0, score_fn=score_fn)

    # Assert the shapes of the outputs
    assert xs.shape == (100, 2)
    assert ts.shape == (100,)


def test_diffuser1d_get_trajectory_generator():
    seed = 0
    sde = SDE(name="brownian", sigma=1.0)
    dt = 2e-2
    diffusion = Diffuser1D(seed, sde, dt)

    x0 = jnp.zeros(32)
    batch_size = 64
    generator = diffusion.get_trajectory_generator(x0, batch_size)

    xss, tss, gradss, *weightss = next(generator)

    # Assert the shapes of the outputs
    assert xss.shape == (batch_size, 50, 32)
    assert tss.shape == (batch_size, 50)
    assert gradss.shape == (batch_size, 50, 32)

    loss = diffusion.dsm_loss(xss, gradss, *weightss)
    assert loss.shape == ()
    assert loss.dtype == jnp.float32

def test_diffuser2d_get_trajectory_generator():
    seed = 0
    sde = SDE(name="stochastic_heat", kappa=0.1, sigma=0.1, dx=0.2)
    dt = 0.1
    diffuser = Diffuser2D(seed, sde, dt)

    x0 = None
    batch_size = 8
    generator = diffuser.get_trajectory_generator(
        x0=x0,
        batch_size=batch_size,
        file_path="/home/gefan/Projects/scoreoperator/src/data/raw/heat_boundary_res128x128_50tsteps_2500_train.h5"
    )

    xss, tss, gradss, *args = next(iter(generator))
    assert xss.shape == (batch_size, 50, 32, 32)
    assert tss.shape == (batch_size, 50)
    assert gradss.shape == (batch_size, 50, 32, 32)

    loss = diffuser.dsm_loss(xss, gradss, *args)
    assert loss.shape == ()
    assert loss.dtype == jnp.float32

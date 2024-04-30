from jax import random
import jax.numpy as jnp

from ..models.diffusion.diffuser import Diffuser
from ..models.diffusion.sde import BrownianSDE

def test_diffusion_solve_sde():
    seed = 0
    dim = 2
    sde = BrownianSDE(dim=dim, sigma=1.0)
    dt = 1e-2
    diffusion = Diffusion(seed, sde, dt)

    rng_key = random.PRNGKey(seed)
    x0 = jnp.array([0.0, 0.0])
    xs, ts, grads = diffusion.solve_sde(rng_key, x0)

    # Assert the shapes of the outputs
    assert xs.shape == (100, 2)
    assert ts.shape == (100,)
    assert grads.shape == (99, 2)

def test_diffusion_solve_reverse_bridge_sde():
    seed = 0
    dim = 2
    sde = BrownianSDE(dim=dim, sigma=1.0)
    dt = 1e-2
    diffusion = Diffusion(seed, sde, dt)

    rng_key = random.PRNGKey(seed)
    x0 = jnp.array([0.0, 0.0])
    score_fn = lambda t, x: x / (1.0 - t)
    xs, ts = diffusion.solve_reverse_bridge_sde(rng_key, x0, score_fn=score_fn)

    # Assert the shapes of the outputs
    assert xs.shape == (100, 2)
    assert ts.shape == (100,)


def test_diffusion_get_trajectory_generator():
    seed = 0
    sde = BrownianSDE(sigma=1.0)
    dt = 1e-2
    diffusion = Diffuser(seed, sde, dt)

    x0 = jnp.zeros(32)
    batch_size = 64
    # generator = diffusion.get_trajectory_generator(x0, batch_size)
    generator = diffusion.get_trajectory_generator(x0, batch_size)

    # xss, tss, covss, gradss = next(generator)

    # # Assert the shapes of the outputs
    # assert xss.shape == (batch_size, 100, dim)
    # assert tss.shape == (batch_size, 100)
    # assert gradss.shape == (batch_size, 100, dim)


    for _ in range(100):
        print("Iteration")
        xss, tss, covss, gradss = next(generator)

        # Assert the shapes of the outputs
        assert xss.shape == (batch_size, 100, 32)
        assert tss.shape == (batch_size, 100)
        assert gradss.shape == (batch_size, 100, 32)

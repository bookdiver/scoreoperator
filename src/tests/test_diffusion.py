import jax
import jax.numpy as jnp

from ..data.toys.toy_shapes import Circle
from ..models.diffusion.sde import BrownianSDE, VPSDE
from ..models.diffusion.gaussian_process import GaussianProcess
from ..models.diffusion.diffusion import DiffusionLoader

def test_brownian_diffusion():
    gp = GaussianProcess(input_dim=1, output_dim=2, n_sample_pts=16, kernel_type='delta', sigma=1.0)
    sde = BrownianSDE(gp, sigma=1.0)
    init_cond = Circle().sample(n_pts=16)
    loader = DiffusionLoader(sde, seed=0, init_cond=init_cond, batch_size=4, shuffle=True)
    xs, ts, zs = next(loader)
    assert xs.shape == (4, 16, 2)
    assert ts.shape == (4, )
    assert zs.shape == (4, 16, 2)

    xs_next, ts_next, zs_next = next(loader)
    assert jnp.all(xs != xs_next)
    assert jnp.all(ts != ts_next)
    assert jnp.all(zs != zs_next)
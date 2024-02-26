import pytest
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from ..random_fields import GaussianRandomField

def test_random_fields():
    sigma = 1.0
    alpha = 2.0
    tau = 1.0
    size = 64
    key = jax.random.PRNGKey(42)

    igrf = GaussianRandomField(dim=2, size=size, gf_type='igrf', sigma=sigma)
    grf = GaussianRandomField(dim=2, size=size, gf_type='grf', sigma=sigma, alpha=alpha, tau=tau)
    pgrf = GaussianRandomField(dim=2, size=size, gf_type='pgrf', sigma=sigma, alpha=alpha, tau=tau)

    n_samples = 3
    z_igrf = igrf.sample(n_samples, key)
    z_grf = grf.sample(n_samples, key)
    z_pgrf = pgrf.sample(n_samples, key)
    assert z_igrf.shape == (n_samples, size, size)
    assert z_grf.shape == (n_samples, size, size)
    assert z_pgrf.shape == (n_samples, size, size)
    assert z_igrf.dtype == jnp.float32
    assert z_grf.dtype == jnp.float32
    assert z_pgrf.dtype == jnp.float32

    fig, ax = plt.subplots(3, 3)
    for i in range(3):
        im1 = ax[i, 0].imshow(z_igrf[i])
        im2 = ax[i, 1].imshow(z_grf[i])
        im3 = ax[i, 2].imshow(z_pgrf[i])
        fig.colorbar(im1, ax=ax[i, 0])
        fig.colorbar(im2, ax=ax[i, 1])
        fig.colorbar(im3, ax=ax[i, 2])
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
        ax[i, 2].set_axis_off()
    fig.savefig('./plot_results/test_random_fields.png')

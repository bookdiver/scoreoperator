import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from ..models.diffusion.gaussian_process import GaussianProcess


def test_gp_1d():
    sigma = 1.0
    nu = 2.5
    kappa = 0.5
    n_sample_pts = 64
    key = jax.random.PRNGKey(42)

    igrf = GaussianProcess(input_dim=1, output_dim=1, n_sample_pts=n_sample_pts, kernel_type='delta', sigma=sigma)
    mgrf = GaussianProcess(input_dim=1, output_dim=1, n_sample_pts=n_sample_pts, kernel_type='matern', sigma=sigma, nu=nu, kappa=kappa)
    ggrf = GaussianProcess(input_dim=1, output_dim=1, n_sample_pts=n_sample_pts, kernel_type='gaussian', sigma=sigma, kappa=kappa)

    n_samples = 3
    z_igrf = igrf.sample_batch(key, n_samples)
    z_mgrf = mgrf.sample_batch(key, n_samples)
    z_ggrf = ggrf.sample_batch(key, n_samples)
    assert z_igrf.shape == (n_samples, n_sample_pts, 1)
    assert z_mgrf.shape == (n_samples, n_sample_pts, 1)
    assert z_ggrf.shape == (n_samples, n_sample_pts, 1)
    assert z_igrf.dtype == jnp.float32
    assert z_mgrf.dtype == jnp.float32
    assert z_ggrf.dtype == jnp.float32

    fig, ax = plt.subplots(3, 3)
    for i in range(3):
        ax[i, 0].plot(z_igrf[i, :, 0])
        ax[i, 1].plot(z_mgrf[i, :, 0])
        ax[i, 2].plot(z_ggrf[i, :, 0])
    plt.show()

def test_gp_2d():
    sigma = 1.0
    nu = 2.5
    kappa = 0.1
    n_sample_pts = 64
    key = jax.random.PRNGKey(42)

    igrf = GaussianProcess(input_dim=2, output_dim=1, n_sample_pts=n_sample_pts, kernel_type='delta', sigma=sigma)
    mgrf = GaussianProcess(input_dim=2, output_dim=1, n_sample_pts=n_sample_pts, kernel_type='matern', sigma=sigma, nu=nu, kappa=kappa)
    ggrf = GaussianProcess(input_dim=2, output_dim=1, n_sample_pts=n_sample_pts, kernel_type='gaussian', sigma=sigma, kappa=kappa)

    n_samples = 3
    z_igrf = igrf.sample_batch(key, n_samples)
    z_mgrf = mgrf.sample_batch(key, n_samples)
    z_ggrf = ggrf.sample_batch(key, n_samples)
    assert z_igrf.shape == (n_samples, n_sample_pts, n_sample_pts, 1)
    assert z_mgrf.shape == (n_samples, n_sample_pts, n_sample_pts, 1)
    assert z_ggrf.shape == (n_samples, n_sample_pts, n_sample_pts, 1)
    assert z_igrf.dtype == jnp.float32
    assert z_mgrf.dtype == jnp.float32
    assert z_ggrf.dtype == jnp.float32

    fig, ax = plt.subplots(3, 3)
    for i in range(3):
        im1 = ax[i, 0].imshow(z_igrf[i, :, :, 0])
        im2 = ax[i, 1].imshow(z_mgrf[i, :, :, 0])
        im3 = ax[i, 2].imshow(z_ggrf[i, :, :, 0])
        fig.colorbar(im1, ax=ax[i, 0])
        fig.colorbar(im2, ax=ax[i, 1])
        fig.colorbar(im3, ax=ax[i, 2])
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
        ax[i, 2].set_axis_off()
    plt.show()
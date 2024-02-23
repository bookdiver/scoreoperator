import pytest
import matplotlib.pyplot as plt
import jax

from ..random_fields import *

def test_random_fields():
    sigma = 1.0
    alpha = 2.0
    tau = 3.0
    size = (64, 64)
    key = jax.random.PRNGKey(42)

    igrf = IndependentGRF2D(size, sigma)
    grf = GRF2D(size, sigma, alpha, tau)
    pgrf = PeriodicGRF2D(size, sigma, alpha, tau)

    n_samples = 3
    z_igrf = igrf.sample(key, n_samples)
    z_grf = grf.sample(key, n_samples)
    z_pgrf = pgrf.sample(key, n_samples)
    assert z_igrf.shape == (n_samples, *size, 2)
    assert z_grf.shape == (n_samples, *size, 2)
    assert z_pgrf.shape == (n_samples, *size, 2)
    assert z_igrf.dtype == jnp.float32
    assert z_grf.dtype == jnp.float32
    assert z_pgrf.dtype == jnp.float32

    fig1, ax1 = plt.subplots(3, 3)
    for i in range(3):
        im1 = ax1[i, 0].imshow(z_igrf[i, :, :, 0])
        im2 = ax1[i, 1].imshow(z_grf[i, :, :, 0])
        im3 = ax1[i, 2].imshow(z_pgrf[i, :, :, 0])
        fig1.colorbar(im1, ax=ax1[i, 0])
        fig1.colorbar(im2, ax=ax1[i, 1])
        fig1.colorbar(im3, ax=ax1[i, 2])
        ax1[i, 0].set_axis_off()
        ax1[i, 1].set_axis_off()
        ax1[i, 2].set_axis_off()
    fig1.savefig('./plot_results/test_random_fields_1.png')

    fig2, ax2 = plt.subplots(3, 3)
    for i in range(3):
        im1 = ax2[i, 0].imshow(z_igrf[i, :, :, 1])
        im2 = ax2[i, 1].imshow(z_grf[i, :, :, 1])
        im3 = ax2[i, 2].imshow(z_pgrf[i, :, :, 1])
        fig2.colorbar(im1, ax=ax2[i, 0])
        fig2.colorbar(im2, ax=ax2[i, 1])
        fig2.colorbar(im3, ax=ax2[i, 2])
        ax2[i, 0].set_axis_off()
        ax2[i, 1].set_axis_off()
        ax2[i, 2].set_axis_off()
    fig2.savefig('./plot_results/test_random_fields_2.png')
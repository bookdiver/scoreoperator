import jax.numpy as jnp
from ..models.diffusion.sde import BrownianSDE

def test_brownian_sde():
    dim = 2
    sigma = 1.0
    sde = BrownianSDE(dim, sigma)

    t = 0.0
    x = jnp.array([1.0, 2.0])

    # Test f function
    f_result = sde.f(t, x)
    assert jnp.all(f_result == jnp.zeros_like(x))

    # Test g function
    g_result = sde.g(t, x)
    expected_g_result = sigma * jnp.eye(dim)
    assert jnp.all(g_result == expected_g_result)

    # Test covariance function
    covariance_result = sde.covariance(t, x)
    expected_covariance_result = sigma**2 * jnp.eye(dim)
    assert jnp.all(covariance_result == expected_covariance_result)

    # Test inv_covariance function
    inv_covariance_result = sde.inv_covariance(t, x)
    expected_inv_covariance_result = 1.0 / sigma**2 * jnp.eye(dim)
    assert jnp.all(inv_covariance_result == expected_inv_covariance_result)

    # Test get_reverse_bridge function
    def score_fn(t: float, x: jnp.ndarray):
        return x / (1. - t)
    reverse_sde = sde.get_reverse_bridge(score_fn)

    # Test f function of reverse_sde
    reverse_f_result = reverse_sde.f(t, x)
    expected_reverse_f_result = sde.covariance(t, x) @ score_fn(t, x)
    assert jnp.all(reverse_f_result == expected_reverse_f_result)

    # Test g function of reverse_sde
    reverse_g_result = reverse_sde.g(t, x)
    expected_reverse_g_result = sde.g(t, x)
    assert jnp.all(reverse_g_result == expected_reverse_g_result)
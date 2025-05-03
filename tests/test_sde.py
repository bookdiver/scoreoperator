import pytest
import jax.numpy as jnp
import numpy as np
from unittest.mock import MagicMock

from scoreoperator.diffusion.sde import BaseSDE, ProjectedSDE, CylindricalBrownianSDE
from scoreoperator.utils.data import BaseData

# Fixtures
@pytest.fixture
def mock_data():
    mock = MagicMock(spec=BaseData)
    mock.eval.return_value = jnp.zeros((10, 2))
    return mock

@pytest.fixture
def base_config(mock_data):
    return {
        'T': 1.0,
        'x0': mock_data,
        'xT': mock_data
    }

@pytest.fixture
def cylindrical_config(base_config):
    config = base_config.copy()
    config['sigma'] = 0.1
    return config

class TestSDE:
    def test_projected_sde_init(self, mock_data):
        """Test ProjectedSDE initialization and properties"""
        # Mock functions
        f = lambda t, x: jnp.zeros_like(x)
        g = lambda t, x: None
        apply_g = lambda Phi, dW: 0.1 * dW
        
        # Initialize ProjectedSDE
        t_shape = (100,)
        x_shape = (10, 2)
        w_shape = (10, 2)
        
        projected_sde = ProjectedSDE(f, g, apply_g, mock_data, mock_data, 1.0, t_shape, x_shape, w_shape)
        
        # Test attributes
        assert projected_sde.T == 1.0
        assert projected_sde.t_shape == t_shape
        assert projected_sde.x_shape == x_shape
        assert projected_sde.w_shape == w_shape
        
        # Test properties
        assert projected_sde.ts.shape == t_shape
        assert np.isclose(projected_sde.dt, 0.01)  # 1.0 / 100
        
        # Test data projection
        assert projected_sde.x0_.shape == x_shape
        assert projected_sde.xT_.shape == x_shape

    def test_cylindrical_brownian_sde(self, cylindrical_config):
        """Test CylindricalBrownianSDE initialization and methods"""
        sde = CylindricalBrownianSDE(cylindrical_config)
        
        # Test attributes
        assert sde.T == 1.0
        assert sde.sigma == 0.1
        
        # Test methods
        x = jnp.ones((10, 2))
        t = 0.5
        
        # Test f method (drift)
        f_result = sde.f(t, x)
        assert f_result.shape == x.shape
        assert jnp.all(f_result == 0)
        
        # Test g method
        g_result = sde.g(t, x)
        assert g_result is None
        
        # Test apply_g method
        dW = jnp.ones((10, 2))
        apply_g_result = sde.apply_g(None, dW)
        assert apply_g_result.shape == dW.shape
        assert jnp.all(apply_g_result == 0.1)

    def test_base_sde_project(self, cylindrical_config, mock_data):
        """Test BaseSDE.project method"""
        # Use CylindricalBrownianSDE as a concrete subclass for testing
        sde = CylindricalBrownianSDE(cylindrical_config)
        
        t_shape = (100,)
        x_shape = (10, 2)
        w_shape = (10, 2)
        
        projected = sde.project(x_shape, t_shape, w_shape)
        
        # Verify it's a ProjectedSDE instance
        assert isinstance(projected, ProjectedSDE)
        assert projected.T == sde.T
        assert projected.t_shape == t_shape
        assert projected.x_shape == x_shape
        assert projected.w_shape == w_shape

    def test_base_sde_get_reverse_bridge(self, cylindrical_config):
        """Test BaseSDE.get_reverse_bridge static method"""
        # Use CylindricalBrownianSDE as a concrete subclass for testing
        forward_sde = CylindricalBrownianSDE(cylindrical_config)
        
        # Mock score function
        score = lambda t, x: jnp.zeros_like(x)
        
        reverse_sde = BaseSDE.get_reverse_bridge(forward_sde, score)
        
        # Test if reverse SDE has correct attributes
        assert reverse_sde.T == forward_sde.T
        assert reverse_sde.x0 is forward_sde.xT
        assert reverse_sde.xT is forward_sde.x0
        
        # Test if reverse SDE f method correctly reverses the forward SDE
        x = jnp.ones((10, 2))
        t = 0.5
        
        reverse_f = reverse_sde.f(t, x)
        expected_f = -forward_sde.f(forward_sde.T - t, x) + score(forward_sde.T - t, x)
        assert jnp.allclose(reverse_f, expected_f) 
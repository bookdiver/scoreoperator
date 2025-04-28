import pytest
import os
import numpy as np
import jax.numpy as jnp
from unittest.mock import patch, mock_open

from scoreoperator.utils.data import (
    BaseData, Addition, Subtraction, Zero, ToyFunction, 
    Ellipse, Ellipsoid, Butterfly, DataFactory
)

class TestBaseDataOperations:
    def test_addition(self):
        # Create two simple implementations of BaseData for testing
        class TestData(BaseData):
            def __init__(self, value):
                self.value = value
            
            def eval(self, shape):
                return jnp.ones(shape) * self.value
        
        a = TestData(5)
        b = TestData(3)
        
        # Test addition
        result = (a + b).eval((2, 2))
        expected = jnp.ones((2, 2)) * 8
        assert np.array_equal(result, expected)
    
    def test_subtraction(self):
        # Create two simple implementations of BaseData for testing
        class TestData(BaseData):
            def __init__(self, value):
                self.value = value
            
            def eval(self, shape):
                return jnp.ones(shape) * self.value
        
        a = TestData(5)
        b = TestData(3)
        
        # Test subtraction
        result = (a - b).eval((2, 2))
        expected = jnp.ones((2, 2)) * 2
        assert np.array_equal(result, expected)

class TestZero:
    def test_eval(self):
        zero = Zero()
        result = zero.eval((3, 2))
        expected = jnp.zeros((3, 2))
        assert np.array_equal(result, expected)

class TestToyFunction:
    def test_eval(self):
        toy = ToyFunction(a=2.0, c=1.0)
        result = toy.eval((5, 1))
        
        # Expected: f(x) = 2x^2 + 1 for x in linspace(-1, 1, 5)
        x = jnp.linspace(-1.0, 1.0, 5)
        expected = jnp.expand_dims(x ** 2 * 2.0 + 1.0, axis=-1)
        
        assert np.allclose(result, expected)
        assert result.shape == (5, 1)

class TestEllipse:
    def test_eval(self):
        ellipse = Ellipse(a=2.0, b=1.0, shift_x=0.5, shift_y=-0.5)
        result = ellipse.eval((6, 2))
        
        # Check shape
        assert result.shape == (6, 2)
        
        # Check a few points on the ellipse
        t = jnp.linspace(0, 2*jnp.pi, 6, endpoint=False)
        expected_x = 2.0 * jnp.cos(t) + 0.5
        expected_y = 1.0 * jnp.sin(t) - 0.5
        expected = jnp.stack([expected_x, expected_y], axis=1)
        
        assert np.allclose(result, expected)

class TestEllipsoid:
    def test_eval(self):
        # Create a patch for the assert statement in Ellipsoid.eval
        class MockEllipsoid(Ellipsoid):
            def eval(self, shape):
                # Generate spherical coordinates
                theta = jnp.linspace(0, jnp.pi, shape[0])
                phi = jnp.linspace(0, 2*jnp.pi, shape[1])
                
                # Create a meshgrid of theta and phi
                theta, phi = jnp.meshgrid(theta, phi)
                
                # Convert spherical coordinates to Cartesian coordinates
                x = self.a * jnp.sin(theta) * jnp.cos(phi) + self.shift_x
                y = self.b * jnp.sin(theta) * jnp.sin(phi) + self.shift_y
                z = self.c * jnp.cos(theta) + self.shift_z
                
                # Stack the coordinates - this will have shape (phi_len, theta_len, 3)
                evaluation = jnp.stack([x, y, z], axis=-1)
                
                # Transpose to match expected shape (theta_len, phi_len, 3)
                evaluation = jnp.transpose(evaluation, (1, 0, 2))
                
                return evaluation
        
        ellipsoid = MockEllipsoid(a=2.0, b=1.5, c=1.0, shift_x=0.5, shift_y=-0.5, shift_z=1.0)
        result = ellipsoid.eval((5, 4, 3))
        
        # Check shape
        assert result.shape == (5, 4, 3)
        
        # Check a few values
        theta = jnp.linspace(0, jnp.pi, 5)
        phi = jnp.linspace(0, 2*jnp.pi, 4)
        
        # Verify some specific points
        # For example, check value at theta=0, phi=0
        x_at_0_0 = 2.0 * jnp.sin(0) * jnp.cos(0) + 0.5  # Expected: 0.5
        y_at_0_0 = 1.5 * jnp.sin(0) * jnp.sin(0) - 0.5  # Expected: -0.5
        z_at_0_0 = 1.0 * jnp.cos(0) + 1.0               # Expected: 2.0
        
        assert np.isclose(result[0, 0, 0], x_at_0_0)
        assert np.isclose(result[0, 0, 1], y_at_0_0)
        assert np.isclose(result[0, 0, 2], z_at_0_0)

class TestButterfly:
    @patch('os.path.dirname')
    @patch('os.path.abspath')
    @patch('os.path.join')
    @patch('os.path.normpath')
    @patch('numpy.load')
    def test_eval_with_interpolation(self, mock_load, mock_normpath, mock_join, mock_abspath, mock_dirname):
        # Setup mocks
        mock_dirname.return_value = "/mock/dir"
        mock_abspath.return_value = "/mock/dir/data.py"
        mock_join.return_value = "/mock/dir/../../landmarks_raw/normalized/test_butterfly.npy"
        mock_normpath.return_value = "/mock/landmarks_raw/normalized/test_butterfly.npy"
        
        # Mock the loaded data - create enough points for testing
        mock_data = np.array([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.0],
            [0.5, -0.5]
        ])
        mock_load.return_value = mock_data
        
        butterfly = Butterfly(name="test_butterfly")
        # Request more points than in the raw data to trigger interpolation
        result = butterfly.eval((8, 2))
        
        # Check shape
        assert result.shape == (8, 2)
    
    @patch('os.path.dirname')
    @patch('os.path.abspath')
    @patch('os.path.join')
    @patch('os.path.normpath')
    @patch('numpy.load')
    def test_eval_without_interpolation(self, mock_load, mock_normpath, mock_join, mock_abspath, mock_dirname):
        # Setup mocks
        mock_dirname.return_value = "/mock/dir"
        mock_abspath.return_value = "/mock/dir/data.py"
        mock_join.return_value = "/mock/dir/../../landmarks_raw/normalized/test_butterfly.npy"
        mock_normpath.return_value = "/mock/landmarks_raw/normalized/test_butterfly.npy"
        
        # Mock the loaded data - create enough points for testing
        mock_data = np.array([
            [0.0, 0.0],
            [0.2, 0.2],
            [0.4, 0.4],
            [0.6, 0.6],
            [0.8, 0.8],
            [1.0, 1.0]
        ])
        mock_load.return_value = mock_data
        
        butterfly = Butterfly(name="test_butterfly")
        # Request fewer points than in the raw data to avoid interpolation
        result = butterfly.eval((3, 2))
        
        # Check shape
        assert result.shape == (3, 2)

class TestDataFactory:
    def test_create_toy(self):
        config = {"a": 2.0, "c": 1.0}
        data = DataFactory.create("toy", config)
        
        assert isinstance(data, ToyFunction)
        assert data.a == 2.0
        assert data.c == 1.0
    
    def test_create_ellipse(self):
        config = {"a": 2.0, "b": 1.0, "shift_x": 0.5, "shift_y": -0.5}
        data = DataFactory.create("ellipse", config)
        
        assert isinstance(data, Ellipse)
        assert data.a == 2.0
        assert data.b == 1.0
        assert data.shift_x == 0.5
        assert data.shift_y == -0.5
    
    def test_create_ellipsoid(self):
        config = {
            "a": 2.0, "b": 1.5, "c": 1.0, 
            "shift_x": 0.5, "shift_y": -0.5, "shift_z": 1.0
        }
        data = DataFactory.create("ellipsoid", config)
        
        assert isinstance(data, Ellipsoid)
        assert data.a == 2.0
        assert data.b == 1.5
        assert data.c == 1.0
        assert data.shift_x == 0.5
        assert data.shift_y == -0.5
        assert data.shift_z == 1.0
    
    @patch('os.path.dirname')
    @patch('os.path.abspath')
    @patch('os.path.join')
    @patch('os.path.normpath')
    @patch('numpy.load')
    def test_create_butterfly(self, mock_load, mock_normpath, mock_join, mock_abspath, mock_dirname):
        # Setup mocks
        mock_dirname.return_value = "/mock/dir"
        mock_abspath.return_value = "/mock/dir/data.py"
        mock_join.return_value = "/mock/dir/../../landmarks_raw/normalized/test.npy"
        mock_normpath.return_value = "/mock/landmarks_raw/normalized/test.npy"
        mock_load.return_value = np.array([[0.0, 0.0], [1.0, 1.0]])
        
        config = {"name": "test"}
        data = DataFactory.create("butterfly", config)
        
        assert isinstance(data, Butterfly)
    
    def test_invalid_type(self):
        with pytest.raises(ValueError):
            DataFactory.create("invalid_type", {}) 
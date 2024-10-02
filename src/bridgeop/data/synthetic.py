from typing import Callable

import jax.numpy as jnp
from jax.random import uniform, normal, PRNGKey

from .function import FunctionData

class QuadraticData(FunctionData):
    """ Quadratic function: f(x) = ax^2 + c, a function R->R
    """
    do_dim: int = 1
    co_dim: int = 1
    # eps: float = 1e-4           # variance of the Gaussian noise
    eps: float = 1e-2           # magnitude of the uniform distributed noise

    def __init__(self, a: float = 1.0, shift: float = 0.0, eps: float = None):
        self.a = a
        self.shift = shift
        self.eps = eps if eps is not None else self.eps

    def _noise_fn(self, rng_key: PRNGKey) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """ Add Gaussian/Uniform noise to function data
        """
        # return lambda data: normal(rng_key, shape=data.shape) * jnp.sqrt(self.eps)
        return lambda data: uniform(rng_key, shape=data.shape, minval=-self.eps, maxval=self.eps)

    def _add_noise(self, data: jnp.ndarray, rng_key: PRNGKey) -> jnp.ndarray:
        noise_fn = self._noise_fn(rng_key)
        return data + noise_fn(data)
    
    def _eval(self, n: int) -> jnp.ndarray:
        return jnp.expand_dims(jnp.linspace(-1.0, 1.0, n) ** 2 * self.a + self.shift, axis=-1)

class Ellipse:
    """ Ellipse shape data, which is treated as a function R->R^2, i.e. parametric equations"""
    a: float
    b: float
    shift_x: float
    shift_y: float

    def __init__(self, a: float = 1.0, b: float = 1.0, shift_x: float = 0.0, shift_y: float = 0.0):
        self.a = a
        self.b = b
        self.shift_x = shift_x
        self.shift_y = shift_y
    
    def eval(self, n_pts: int) -> jnp.ndarray:
        t = jnp.linspace(0, 2*jnp.pi, n_pts, endpoint=False)
        x = self.a * jnp.cos(t) + self.shift_x
        y = self.b * jnp.sin(t) + self.shift_y
        return jnp.stack([x, y], axis=1)
    
class Ellipsoid:
    """ Ellipsoid shape data, which is treated as a function R->R^3"""
    a: float
    b: float
    c: float
    shift_x: float
    shift_y: float
    shift_z: float
    
    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 1.0, shift_x: float = 0.0, shift_y: float = 0.0, shift_z: float = 0.0):
        self.a = a
        self.b = b
        self.c = c
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.shift_z = shift_z
        
    def eval(self, n_pts: int) -> jnp.ndarray:
        points = []
        phi = jnp.pi * (3. - jnp.sqrt(5.))  # golden angle in radians
        for i in range(n_pts):
            y = 1 - (i / float(n_pts - 1)) * 2  # y goes from 1 to -1
            r = jnp.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment
            x = jnp.cos(theta) * r
            z = jnp.sin(theta) * r

            points.append((self.a * x + self.shift_x, 
                           self.b * y + self.shift_y, 
                           self.c * z + self.shift_z))

        return jnp.array(points)
    
    
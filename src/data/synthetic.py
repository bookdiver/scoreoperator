from typing import Union
from .function import Function

import jax.numpy as jnp
from jax.random import normal, PRNGKey

class Circle(Function):
    do_dim: int = 1
    co_dim: int = 2

    def __init__(self, r: float = 1.0, shift_x: float = 0.0, shift_y: float = 0.0, eps: float = 1e-2):
        self.r = r
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.eps = eps

    def noise(self, n_samples: int, rng_key: PRNGKey):
        return normal(rng_key, (n_samples, self.co_dim)) * self.eps
    
    def _sample(self, n_samples: int):
        t = jnp.linspace(0, 2*jnp.pi, n_samples, endpoint=False)
        circ_x = self.r * jnp.cos(t) + self.shift_x
        circ_y = self.r * jnp.sin(t) + self.shift_y
        return jnp.stack([circ_x, circ_y], axis=1)

class Quadratic(Function):
    do_dim: int = 1
    co_dim: int = 1

    def __init__(self, a: float = 1.0, shift: float = 0.0, eps: float = 1e-4):
        self.a = a
        self.shift = shift
        self.eps = eps

    def noise(self, n_samples: int, rng_key: PRNGKey):
        return normal(rng_key, (n_samples, self.co_dim)) * self.eps

    def _sample(self, n_samples: int):
        return jnp.expand_dims(jnp.linspace(-1.0, 1.0, n_samples) ** 2 * self.a + self.shift, axis=-1)

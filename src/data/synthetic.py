from typing import Union
from .function import Function

import jax.numpy as jnp

class Zero(Function):
    do_dim: int = 1
    co_dim: int = 1

    def _sample(self, n_samples: int):
        return jnp.zeros((n_samples, self.co_dim))

class Circle(Function):
    do_dim: int = 1
    co_dim: int = 2

    def __init__(self, r: Union[float, str] = 1.0, shift_x = 0.0, shift_y: float = 0.0):
        self.r = float(r)
        self.shift_x = shift_x
        self.shift_y = shift_y
    
    def _sample(self, n_samples: int):
        t = jnp.linspace(0, 2*jnp.pi, n_samples, endpoint=False)
        circ_x = self.r * jnp.cos(t) + self.shift_x
        circ_y = self.r * jnp.sin(t) + self.shift_y
        return jnp.stack([circ_x, circ_y], axis=1)

class Quadratic(Function):
    do_dim: int = 1
    co_dim: int = 1

    def __init__(self, a: Union[float, str]=1.0):
        self.a = float(a)

    def _sample(self, n_samples: int):
        return jnp.linspace(-1.0, 1.0, n_samples) ** 2 * self.a

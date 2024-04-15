from .shape import Shape

import jax.numpy as jnp

class Circle(Shape):
    def __init__(self, r: float=1.0, shift_x: float=0.0, shift_y: float=0.0):
        self.r = r
        self.shift_x = shift_x
        self.shift_y = shift_y
    
    def sample(self, n_samples: int):
        t = jnp.linspace(0, 2*jnp.pi, n_samples, endpoint=False)
        circ_x = self.r * jnp.cos(t) + self.shift_x
        circ_y = self.r * jnp.sin(t) + self.shift_y
        return jnp.stack([circ_x, circ_y], axis=1)
        
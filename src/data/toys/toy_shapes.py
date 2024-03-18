import jax.numpy as jnp

class Circle:
    def __init__(self, radius=1.0, shift_x=0.0, shift_y=0.0):
        self._radius = radius
        self._shift_x = shift_x
        self._shift_y = shift_y
    
    def sample(self, n_pts):
        circ_x = self._radius * jnp.cos(jnp.linspace(0, 2*jnp.pi, n_pts, endpoint=False)) + self._shift_x
        circ_y = self._radius * jnp.sin(jnp.linspace(0, 2*jnp.pi, n_pts, endpoint=False)) + self._shift_y
        return jnp.stack([circ_x, circ_y], axis=1)

class TwoMoons:
    def __init__(self, radius=1.0, shift_x=0.0, shift_y=0.0):
        self._radius = radius
        self._shift_x = shift_x
        self._shift_y = shift_y
    
    def _up_circle(self, n_pts):
        circ_x = self._radius * jnp.cos(jnp.linspace(0, jnp.pi, n_pts)) + self._shift_x / 2.0
        circ_y = self._radius * jnp.sin(jnp.linspace(0, jnp.pi, n_pts)) - self._shift_y / 2.0
        return jnp.stack([circ_x, circ_y], axis=1)
    
    def _down_circle(self, n_pts):
        circ_x = self._radius * jnp.cos(jnp.linspace(0, jnp.pi, n_pts)) - self._shift_x / 2.0
        circ_y = -self._radius * jnp.sin(jnp.linspace(0, jnp.pi, n_pts)) + self._shift_y / 2.0
        return jnp.stack([circ_x, circ_y], axis=1)

    def sample(self, n_pts):
        half_pts = n_pts // 2
        _up_circ = self._up_circle(half_pts)
        _down_circ = self._down_circle(half_pts)
        pts = jnp.concatenate([_up_circ, _down_circ], axis=0)
        return pts
        
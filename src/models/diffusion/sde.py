from __future__ import annotations

import abc
from typing import Callable
import jax.numpy as jnp

from ...data.shape import Shape

class SDE(abc.ABC):

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        pass

    @abc.abstractmethod
    def g(self, t: float, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        pass

    def g2(self, t: float, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        g = self.g(t, x, **kwargs)
        return jnp.dot(g, g.T)
    
    def get_reverse_bridge(self, score_fn: Callable[[float, jnp.ndarray], jnp.ndarray] = None) -> SDE:
        f = self.f
        g = self.g
        g2 = self.g2

        class ReverseSDE(SDE):
            def __init__(self):
                super().__init__()

            def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
                reversed_t = 1.0 - t
                inv_g = jnp.linalg.inv(g(t=reversed_t, x=x).T) / jnp.sqrt(reversed_t)
                score = jnp.dot(inv_g, score_fn(t=reversed_t, x=x))
                return -f(t=reversed_t, x=x) + jnp.dot(g2(t=reversed_t, x=x), score)
            
            def g(self, t: float, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
                reversed_t = 1.0 - t
                return g(t=reversed_t, x=x)
            
            def g2(self, t: float, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
                reversed_t = 1.0 - t
                return g2(t=reversed_t, x=x, **kwargs)
        
        return ReverseSDE()
    
class BrownianSDE(SDE):
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(x)
    
    def g(self, t: float, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self.sigma * jnp.eye(x.shape[-1])
    
    def covariance(self, t: float, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self.sigma**2 * jnp.eye(x.shape[-1])

    def inv_covariance(self, t: float, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return 1.0 / self.sigma**2 * jnp.eye(x.shape[-1])

class EulerianSDE(SDE):
    def __init__(self, sigma: float = 1.0, kappa: float = 0.1, s0: Shape = None):
        super().__init__()
        self.sigma = sigma
        self.kappa = kappa
        self.s0 = s0
    
    def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(x)
    
    def g(self, t: float, x: jnp.ndarray, eps: float = 1e-4) -> jnp.ndarray:
        x = x.reshape(-1, 2)
        n_pts = x.shape[0]
        x = x + self.s0.sample(n_pts)
        kernel_fn = lambda x: self.sigma * jnp.exp(-0.5 * jnp.sum(jnp.square(x), axis=-1) / self.kappa**2)
        dist = x[:, None, :] - x[None, :, :]
        kernel = kernel_fn(dist) + eps * jnp.eye(n_pts)     # Regularization to avoid singular matrix
        Q_half = jnp.einsum("ij,kl->ikjl", kernel, jnp.eye(2))
        Q_half = Q_half.reshape(2*n_pts, 2*n_pts)
        return Q_half
    
        
from __future__ import annotations

import abc
from typing import Tuple

import jax
import jax.numpy as jnp

from ...data.function import Function

class BaseSDE(abc.ABC):

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
    
    def inv_g2(self, t: float, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        g2 = self.g2(t, x, **kwargs)
        return jnp.linalg.inv(g2)
    
    def get_reverse_bridge(self, model) -> BaseSDE:
        f = self.f
        g = self.g
        g2 = self.g2

        if model.matching_obj == "score":
            drift_fn = lambda t, x: jnp.dot(g2(t, x), model(t, x))
        elif model.matching_obj == "gscore":
            drift_fn = lambda t, x: jnp.dot(g(t, x), model(t, x)) 
        elif model.matching_obj == "g2score":
            drift_fn = lambda t, x: model(t, x)
        else:
            raise ValueError(f"Unknown model matching object: {model.matching_obj}")

        class ReverseSDE(BaseSDE):
            def __init__(self):
                super().__init__()

            def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
                reversed_t = 1.0 - t
                return -f(t=reversed_t, x=x) + drift_fn(t=reversed_t, x=x)
            
            def g(self, t: float, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
                reversed_t = 1.0 - t
                return g(t=reversed_t, x=x)
            
            def g2(self, t: float, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
                reversed_t = 1.0 - t
                return g2(t=reversed_t, x=x, **kwargs)
        
        return ReverseSDE()
    
class BrownianSDE(BaseSDE):
    def __init__(self, 
                 sigma: float = 1.0,
                 X0: Function = None):
        super().__init__()
        self.sigma = sigma
        self.X0 = X0

    def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(x)
    
    def g(self, t: float, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self.sigma * jnp.eye(x.shape[-1])
    
    def g2(self, t: float, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self.sigma**2 * jnp.eye(x.shape[-1])

    def inv_g2(self, t: float, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return 1.0 / self.sigma**2 * jnp.eye(x.shape[-1])
    
class OUSDE(BaseSDE):
    def __init__(self, 
                 sigma: float = 1.0, 
                 theta: float = 1.0, 
                 X0: Function = None):
        super().__init__()
        self.sigma = sigma
        self.theta = theta
        self.X0 = X0
    
    def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return -self.theta * x
    
    def g(self, t: float, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self.sigma * jnp.eye(x.shape[-1])
    
    def g2(self, t: float, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self.sigma**2 * jnp.eye(x.shape[-1])
    
    def inv_g2(self, t: float, x: jax.Array, **kwargs) -> jax.Array:
        return 1.0 / self.sigma**2 * jnp.eye(x.shape[-1])

class EulerianSDE(BaseSDE):
    def __init__(self, 
                 sigma: float = 1.0, 
                 kappa: float = 0.1, 
                 X0: Function = None):
        super().__init__()
        self.sigma = sigma
        self.kappa = kappa
        self.X0 = X0
    
    def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(x)
    
    def g(self, t: float, x: jnp.ndarray, eps: float = 1e-4) -> jnp.ndarray:
        x = x.reshape(-1, 2)
        n_pts = x.shape[0]
        x = x + self.X0.sample(n_pts)
        kernel_fn = lambda x: self.sigma * jnp.exp(-0.5 * jnp.sum(jnp.square(x), axis=-1) / self.kappa**2)
        dist = x[:, None, :] - x[None, :, :]
        kernel = kernel_fn(dist) + eps * jnp.eye(n_pts)     # Regularization to avoid singular matrix
        Q_half = jnp.einsum("ij,kl->ikjl", kernel, jnp.eye(2))
        Q_half = Q_half.reshape(2*n_pts, 2*n_pts)
        return Q_half

class EulerianSDELandmarkIndependent(BaseSDE):
    def __init__(self, 
                 sigma: float = 1.0, 
                 kappa: float = 0.1, 
                 X0: Function = None, 
                 grid_sz: int = 50, 
                 grid_range: Tuple[float, float] = (-0.5, 1.5)):
        super().__init__()
        self.sigma = sigma
        self.kappa = kappa
        self.X0 = X0
        self.grid_sz = grid_sz
        self.grid_range = grid_range

    @property
    def noise_dim(self):
        return self.grid_sz**2
    
    @property
    def grid(self):
        grid = jnp.linspace(*self.grid_range, self.grid_sz)
        grid = jnp.stack(jnp.meshgrid(grid, grid, indexing="xy"), axis=-1)
        return grid
    
    def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(x)
    
    def g(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        x = x.reshape(-1, 2)
        n_pts = x.shape[0]
        x = x + self.X0.sample(n_pts)
        kernel_fn = lambda x, y: self.sigma * jnp.exp(-0.5 * jnp.linalg.norm(x-y, axis=-1)**2 / self.kappa**2)
        Q_half = jax.vmap(
        jax.vmap(
                jax.vmap(
                    kernel_fn,
                    in_axes=(None, 0),
                    out_axes=0
                ),
                in_axes=(None, 1),
                out_axes=1
            ),
            in_axes=(0, None),
            out_axes=0
        )(x, self.grid)
        Q_half = Q_half.reshape(n_pts, self.noise_dim)
        Q_half = jnp.einsum("ij,kl->ikjl", Q_half, jnp.eye(2))
        Q_half = Q_half.reshape(2*n_pts, 2*self.noise_dim)
        return Q_half
    
    def g2(self, t: float, x: jnp.ndarray, eps: float = 1e-4) -> jnp.ndarray:
        g = self.g(t, x)
        return jnp.dot(g, g.T) + eps * jnp.eye(g.shape[0])
    
class StochasticHeatEquation(BaseSDE):
    def __init__(self,
                 sigma: float,
                 kappa: float,
                 dx: float):
        self.sigma = sigma
        self.kappa = kappa
        self.dx = dx

    def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return self.kappa * (jnp.roll(x, 1, axis=0) + jnp.roll(x, -1, axis=0) + 
                             jnp.roll(x, 1, axis=1) + jnp.roll(x, -1, axis=1) - 4 * x) / self.dx**2     # Finite difference for Laplacian

    def g(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return self.sigma
    
    def g2(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return self.sigma**2
    
    def inv_g2(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return 1 / self.sigma**2
    
class SDE:
    def __init__(self, name: str, **kwargs):
        if name == "brownian":
            self.sde_ins = BrownianSDE(**kwargs)
        elif name == "ou":
            self.sde_ins = OUSDE(**kwargs)
        elif name == "eulerian":
            self.sde_ins = EulerianSDE(**kwargs)
        elif name == "eulerian_independent":
            self.sde_ins = EulerianSDELandmarkIndependent(**kwargs)
        elif name == "stochastic_heat":
            self.sde_ins = StochasticHeatEquation(**kwargs)
        else:
            raise ValueError(f"Unknown SDE name: {name}")
    
    def __getattr__(self, attr):
        return getattr(self.sde_ins, attr)
    
    def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return self.sde_ins.f(t, x)
    
    def g(self, t: float, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self.sde_ins.g(t, x, **kwargs)
    
    def g2(self, t: float, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self.sde_ins.g2(t, x, **kwargs)
    
    def inv_g2(self, t: float, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self.sde_ins.inv_g2(t, x, **kwargs)
    
    def get_reverse_bridge(self, model) -> BaseSDE:
        return self.sde_ins.get_reverse_bridge(model)
    

from typing import Tuple, Generator, Callable
from functools import partial

import jax
import jax.numpy as jnp

from diffrax import VirtualBrownianTree, MultiTerm, ODETerm, ControlTerm, Euler, SaveAt, diffeqsolve
from .sde import SDE
from .brownian import ReverseVirtualBrownianTree

class Diffuser:
    seed: int
    sde: SDE
    dt: float

    def __init__(
            self, 
            seed: int, 
            sde: SDE, 
            dt: float = 1e-3):
        self.sde = sde
        self.rng_key = jax.random.PRNGKey(seed)
        self.dt = dt
        
        self.reverse_diffusion_bridge_sde = None

    def get_reverse_diffusion_bridge(self, score_fn: Callable[[float, jnp.ndarray], jnp.ndarray]):
        self.reverse_diffusion_bridge_sde = self.sde.get_reverse_bridge(score_fn)
    
    @partial(jax.jit, static_argnums=(0,))
    def solve_sde(self, rng_key: jax.Array, x0: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        brownian = VirtualBrownianTree(
            t0=0.0, t1=1.0, tol=1e-3, shape=(self.sde.dim, ), key=rng_key
        )
        terms = MultiTerm(
            ODETerm(lambda t, y, _: self.sde.f(t, y)),
            ControlTerm(lambda t, y, _: self.sde.g(t, y), brownian)
        )
        solver = Euler()
        saveat = SaveAt(ts=jnp.arange(0.0, 1.0+self.dt, self.dt))
        sol = diffeqsolve(terms, solver, t0=0.0, t1=1.0, dt0=self.dt, saveat=saveat, y0=x0)

        xs, ts = sol.ys, sol.ts
        
        diff_xs = xs[1:] - xs[:-1]
        covs = jax.vmap(lambda t, x: self.sde.covariance(t, x))(ts[:-1], xs[:-1]) / self.dt
        inv_covs = jax.vmap(jnp.linalg.inv)(covs)
        grads = jax.vmap(lambda inv_cov, diff_x: jnp.dot(inv_cov, diff_x))(inv_covs, diff_xs)

        return xs[1:], ts[1:], covs, grads
    
    @partial(jax.jit, static_argnums=(0,), static_argnames=("score_fn",))
    def solve_reverse_bridge_sde(self, rng_key: jax.Array, x0: jnp.ndarray, *, score_fn: Callable[[float, jnp.ndarray], jnp.ndarray]=None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        self.get_reverse_diffusion_bridge(score_fn)

        reverse_brownian = ReverseVirtualBrownianTree(
            t0=0.0, t1=1.0, tol=1e-3, shape=(self.reverse_diffusion_bridge_sde.dim, ), key=rng_key
        )
        terms = MultiTerm(
            ODETerm(lambda t, y, _: self.reverse_diffusion_bridge_sde.f(t, y)),
            ControlTerm(lambda t, y, _: self.reverse_diffusion_bridge_sde.g(t, y), reverse_brownian)
        )
        solver = Euler()
        saveat = SaveAt(ts=jnp.arange(self.dt, 1.0+self.dt, self.dt))
        sol = diffeqsolve(terms, solver, t0=0.0, t1=1.0, dt0=self.dt, saveat=saveat, y0=x0)

        xs, ts = sol.ys, sol.ts
        return xs, ts


    # @partial(jax.jit, static_argnums=(0,))
    def get_trajectory_generator(self, x0: jnp.ndarray, batch_size: int) -> Generator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], None, None]:

        def generator():
            while True:
                xss, tss, covss, gradss = [], [], [], []
                for _ in range(batch_size):
                    self.rng_key, _ = jax.random.split(self.rng_key)
                    xs, ts, covs, grads = self.solve_sde(self.rng_key, x0)
                    xss.append(xs)
                    tss.append(ts)
                    covss.append(covs)
                    gradss.append(grads)
                yield jnp.stack(xss), jnp.stack(tss), jnp.stack(covss), jnp.stack(gradss)
        
        return generator()
        

    


    

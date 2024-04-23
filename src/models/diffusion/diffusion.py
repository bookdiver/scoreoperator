from typing import Tuple, Generator, Callable
from functools import partial

import jax
import jax.numpy as jnp

from diffrax import VirtualBrownianTree, UnsafeBrownianPath, DirectAdjoint, MultiTerm, ODETerm, ControlTerm, Euler, SaveAt, diffeqsolve
from .sde import SDE

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
        # brownian = VirtualBrownianTree(
        #     t0=0.0, t1=1.0, tol=1e-3, shape=(x0.shape[-1], ), key=rng_key
        # )
        brownian = UnsafeBrownianPath(shape=(x0.shape[-1], ), key=rng_key)
        terms = MultiTerm(
            ODETerm(lambda t, y, _: self.sde.f(t, y)),
            ControlTerm(lambda t, y, _: self.sde.g(t, y), brownian)
        )
        solver = Euler()
        saveat = SaveAt(ts=jnp.arange(0.0, 1.0+self.dt, self.dt))
        # sol = diffeqsolve(terms, solver, t0=0.0, t1=1.0, dt0=self.dt, saveat=saveat, y0=x0)
        sol = diffeqsolve(terms, solver, t0=0.0, t1=1.0, dt0=self.dt, saveat=saveat, y0=x0, adjoint=DirectAdjoint())

        xs, ts = sol.ys, sol.ts
        
        diff_xs = xs[1:] - xs[:-1]
        covs_prev = jax.vmap(lambda t, x: self.sde.covariance(t, x))(ts[:-1], xs[:-1])
        covs_now = jax.vmap(lambda t, x: self.sde.covariance(t, x))(ts[1:], xs[1:])
        inv_covs = jax.vmap(jnp.linalg.pinv)(covs_prev)
        grads = jax.vmap(lambda inv_cov, diff_x: jnp.dot(inv_cov, diff_x))(inv_covs, diff_xs) / self.dt

        return xs[1:], ts[1:], covs_now, grads
    
    @partial(jax.jit, static_argnums=(0, 2), static_argnames=("score_fn",))
    def solve_reverse_bridge_sde(self, rng_key: jax.Array, x0: jnp.ndarray, n_steps: int=100, *, score_fn: Callable[[float, jnp.ndarray], jnp.ndarray]=None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        self.get_reverse_diffusion_bridge(score_fn)

        # brownian = VirtualBrownianTree(
        #     t0=0.0, t1=1.0, tol=1e-3, shape=(x0.shape[-1], ), key=rng_key
        # )
        brownian = UnsafeBrownianPath(shape=(x0.shape[-1], ), key=rng_key)
        terms = MultiTerm(
            ODETerm(lambda t, y, _: self.reverse_diffusion_bridge_sde.f(t, y)),
            ControlTerm(lambda t, y, _: self.reverse_diffusion_bridge_sde.g(t, y), brownian)
        )
        solver = Euler()
        saveat = SaveAt(ts=jnp.linspace(0.0, 1.0, 500))
        # sol = diffeqsolve(terms, solver, t0=0.0, t1=1.0, dt0=self.dt, saveat=saveat, y0=x0)
        sol = diffeqsolve(terms, solver, t0=0.0, t1=1.0, dt0=self.dt, saveat=saveat, y0=x0, adjoint=DirectAdjoint())

        xs, ts = sol.ys, sol.ts
        return xs, ts

    def get_trajectory_generator(self, x0: jnp.ndarray, batch_size: int) -> Generator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], None, None]:
        
        solve_sde = jax.vmap(self.solve_sde, in_axes=(0, None))
        def generator():
            while True:
                rng_keys = jax.random.split(self.rng_key, batch_size + 1)
                xss, tss, covss, gradss = solve_sde(rng_keys[1:], x0)
                self.rng_key = rng_keys[0]
                yield xss, tss, covss, gradss
        return generator()

    


    

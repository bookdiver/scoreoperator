from typing import Tuple, Generator
from functools import partial

import jax
import jax.numpy as jnp

from diffrax import (UnsafeBrownianPath, DirectAdjoint, 
                     MultiTerm, ODETerm, 
                     ControlTerm, Euler, SaveAt, diffeqsolve)
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

    def get_reverse_diffusion_bridge(self, model) -> SDE:
        self.reverse_diffusion_bridge_sde = self.sde.get_reverse_bridge(model)

    def get_grads(self, xs: jnp.ndarray, ts: jnp.ndarray, dt: float, noise_scaling: str = "inv_g2") -> jnp.ndarray:
        diff_xs = xs[1:] - xs[:-1] - jax.vmap(lambda t, x: self.sde.f(t, x))(ts[:-1], xs[:-1]) * dt
        if noise_scaling == "inv_g2":
            g2s_tm1 = jax.vmap(lambda t, x: self.sde.g2(t, x))(ts[:-1], xs[:-1])
            scales = jax.vmap(jnp.linalg.inv)(g2s_tm1) 
        elif noise_scaling == "inv_g":
            gs_tm1 = jax.vmap(lambda t, x: self.sde.g(t, x))(ts[:-1], xs[:-1])
            scales = jax.vmap(jnp.linalg.inv)(gs_tm1) 
        elif noise_scaling == "none":
            scales = jax.vmap(lambda x: jnp.eye(x.shape[-1]))(xs[:-1]) 

        grads = jax.vmap(lambda scale, diff_x: jnp.dot(scale, diff_x))(scales, diff_xs) / dt

        return grads

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def solve_sde(self, rng_key: jax.Array, x0: jnp.ndarray, noise_scaling: str = "inv_g2", weighting_output: str = "id") -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if hasattr(self.sde, "noise_dim"):
            noise_dim = self.sde.noise_dim
        else:
            noise_dim = x0.shape[-1]

        brownian = UnsafeBrownianPath(shape=(noise_dim, ), key=rng_key)

        terms = MultiTerm(
            ODETerm(lambda t, y, _: self.sde.f(t, y)),
            ControlTerm(lambda t, y, _: self.sde.g(t, y), brownian)
        )
        solver = Euler()
        saveat = SaveAt(ts=jnp.arange(0.0, 1.0+self.dt, self.dt))
        sol = diffeqsolve(terms, solver, t0=0.0, t1=1.0, dt0=self.dt, saveat=saveat, y0=x0, adjoint=DirectAdjoint())

        xs, ts = sol.ys, sol.ts
        grads = self.get_grads(xs, ts, self.dt, noise_scaling)

        if weighting_output == "g2s":
            g2s = jax.vmap(lambda t, x: self.sde.g2(t, x))(ts[1:], xs[1:])
            return xs[1:], ts[1:], grads, g2s
        elif weighting_output == "inv_g2s":
            inv_g2s = jax.vmap(lambda t, x: self.sde.inv_g2)(ts[1:], xs[1:])
            return xs[1:], ts[1:], grads, inv_g2s
        elif weighting_output == "id":
            return xs[1:], ts[1:], grads, jax.vmap(lambda x: jnp.eye(x.shape[-1]))(xs[1:])
    
    @partial(jax.jit, static_argnums=(0, 2, 3), static_argnames=("model",))
    def solve_reverse_bridge_sde(self, rng_key: jax.Array, x0: jnp.ndarray, ts: jnp.ndarray, *, model = None) -> jnp.ndarray:
        self.get_reverse_diffusion_bridge(model)
        if hasattr(self.sde, "noise_dim"):
            noise_dim = self.sde.noise_dim
        else:
            noise_dim = x0.shape[-1]

        brownian = UnsafeBrownianPath(shape=(noise_dim, ), key=rng_key)
        terms = MultiTerm(
            ODETerm(lambda t, y, _: self.reverse_diffusion_bridge_sde.f(t, y)),
            ControlTerm(lambda t, y, _: self.reverse_diffusion_bridge_sde.g(t, y), brownian)
        )
        solver = Euler()
        saveat = SaveAt(ts=ts)
        sol = diffeqsolve(terms, solver, t0=0.0, t1=1.0, dt0=self.dt, saveat=saveat, y0=x0, adjoint=DirectAdjoint())

        return sol.ys

    def get_trajectory_generator(self, x0: jnp.ndarray, batch_size: int, noise_scaling: str = "inv_g2", weighting_output: str = "g2s") -> Generator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, any], None, None]:
        
        solve_sde = jax.vmap(
            partial(self.solve_sde, noise_scaling=noise_scaling, weighting_output=weighting_output),
            in_axes=(0, None)
        )
        def generator():
            while True:
                rng_keys = jax.random.split(self.rng_key, batch_size + 1)
                xss, tss, gradss, weightss = solve_sde(rng_keys[1:], x0)
                self.rng_key = rng_keys[0]
                yield xss, tss, gradss, weightss
        return generator()
    
    def dsm_loss(self, predss: jnp.ndarray, gradss: jnp.ndarray, weightss: jnp.ndarray = None):
        errorss = predss + gradss
        losses = jnp.einsum("bti,btij,btj->bt", errorss, weightss, errorss)
        losses = jnp.sum(losses, axis=1)
        loss = 0.5 * self.dt * jnp.mean(losses)
        return loss
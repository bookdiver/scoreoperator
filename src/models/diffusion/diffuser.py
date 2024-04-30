from typing import Tuple, Generator
from functools import partial

import jax
import jax.numpy as jnp

from diffrax import (UnsafeBrownianPath, DirectAdjoint, 
                     MultiTerm, ODETerm, 
                     ControlTerm, Euler, SaveAt, diffeqsolve)
from .sde import SDE
from ...utils.util import Approximator

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

    def get_reverse_diffusion_bridge(self, approx: Approximator) -> SDE:
        self.reverse_diffusion_bridge_sde = self.sde.get_reverse_bridge(approx)

    def get_grads(self, xs: jnp.ndarray, ts: jnp.ndarray, dt: float, scaling: str = "g2") -> jnp.ndarray:
        diff_xs = xs[1:] - xs[:-1] - jax.vmap(lambda t, x: self.sde.f(t, x))(ts[:-1], xs[:-1]) * dt
        if scaling == "g2":
            g2s_tm1 = jax.vmap(lambda t, x: self.sde.g2(t, x))(ts[:-1], xs[:-1])
            scales = jax.vmap(jnp.linalg.inv)(g2s_tm1) 
        elif scaling == "g":
            gs_tm1 = jax.vmap(lambda t, x: self.sde.g(t, x))(ts[:-1], xs[:-1])
            scales = jax.vmap(jnp.linalg.inv)(gs_tm1) 
        elif scaling == "none":
            scales = jax.vmap(jnp.eye)(xs.shape[-1]) 

        grads = jax.vmap(lambda scale, diff_x: jnp.dot(scale, diff_x))(scales, diff_xs) / dt

        return grads

    @partial(jax.jit, static_argnums=(0,))
    def solve_sde(self, rng_key: jax.Array, x0: jnp.ndarray, scaling: str = "g2", verbose: str = "none") -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
        grads = self.get_grads(xs, ts, self.dt, scaling)

        if verbose == "g2s":
            g2s = jax.vmap(lambda t, x: self.sde.g2(t, x))(ts[1:], xs[1:])
            return xs[1:], ts[1:], grads, g2s
        elif verbose == "inv_g2s":
            inv_g2s = jax.vmap(lambda t, x: self.sde.inv_g2)(ts[1:], xs[1:])
            return xs[1:], ts[1:], grads, inv_g2s
        elif verbose == "none":
            return xs[1:], ts[1:], grads, None
    
    @partial(jax.jit, static_argnums=(0, 2, 3), static_argnames=("approx",))
    def solve_reverse_bridge_sde(self, rng_key: jax.Array, x0: jnp.ndarray, ts: jnp.ndarray, *, approx: Approximator = None) -> jnp.ndarray:
        self.get_reverse_diffusion_bridge(approx)
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

    def get_trajectory_generator(self, x0: jnp.ndarray, batch_size: int, scaling: str = "g2", verbose: str = "g2s") -> Generator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, any], None, None]:
        
        solve_sde = jax.vmap(
            partial(self.solve_sde, scaling=scaling, verbose=verbose),
            in_axes=(0, None)
        )
        def generator():
            while True:
                rng_keys = jax.random.split(self.rng_key, batch_size + 1)
                xss, tss, gradss, verboses = solve_sde(rng_keys[1:], x0)
                self.rng_key = rng_keys[0]
                yield xss, tss, gradss, verboses
        return generator()
    
    def dsm_loss(self, predss: jnp.ndarray, gradss: jnp.ndarray, weighting: jnp.ndarray = None):
        if weighting is None:
            error = predss + gradss
            losses = jnp.einsum("bti,bti->bt", error, error)
        else:
            losses = jnp.einsum("bti,btij,btj->bt", error, weighting, error)
        losses = jnp.sum(losses, axis=1)
        loss = 0.5 * self.dt * jnp.mean(losses)
        return loss

    


    

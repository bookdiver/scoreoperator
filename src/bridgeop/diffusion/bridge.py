from __future__ import annotations
from typing import Dict, Any, Tuple
from functools import partial

import jax
import jax.numpy as jnp

from .sde import SDEFactory
from ..utils.solver import Wiener, EulerMaruyama, SamplePath

class DiffusionBridge:
    """ The diffuser class for several utilities in the training, including:
        - get reverse diffusion bridge from a defined `model`: self.get_reverse_diffusion_bridge()
        - compute b based on the trajectories: self.get_bs()
        - solve forward SDE: self.solve_forward_sde()
        - solve reverse SDE: self.solve_reverse_sde()
        - get trajectory generator: self.get_trajectory_generator()
        - compute loss: self.dsm_loss()
        
    """

    def __init__(
            self, 
            sde_name: str, 
            sde_kwargs: Dict[str, Any],
            dt: float = 1e-2):
        self.sde = SDEFactory.create(sde_name, **sde_kwargs)
        
        self.wiener = Wiener(sde_kwargs["W_shape"], sde_kwargs["T"], dt=dt)
        self.sde_solver = EulerMaruyama(
            sde=self.sde, 
            wiener=self.wiener
        )
        self.dt = dt

    def get_bs(self, xs: jnp.ndarray, ts: jnp.ndarray) -> jnp.ndarray:
        """ Compute the small step gradient approximation based on the simulated trajectories

        Args:
            xs (jnp.ndarray): simulated trajectories of shape (time_steps, dim)
            ts (jnp.ndarray): time steps of the trajectories of shape (time_steps,)
            scaling (str, optional): scaling factor, can be None, "inv_g", "inv_g2" or None. Defaults to be None.

        Returns:
            jnp.ndarray: _description_
        """
        diff_xs = xs[1:] - xs[:-1] - jax.vmap(lambda t, x: self.sde.f(t, x))(ts[:-1], xs[:-1]) * self.dt
        bs = - diff_xs / self.dt
        return bs

    @partial(jax.jit, static_argnums=(0, 3))
    def solve_forward_sde(self, rng_key: jax.Array, x0: jnp.ndarray, n_batches: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        sol = self.sde_solver.solve(rng_key, x0, n_batches)
        xs, ts = sol.xs, sol.ts
        
        bs = jax.vmap(
            self.get_bs,
            in_axes=(0, None),
            out_axes=0
        )(xs, ts)

        return xs[:, 1:], ts[1:], bs
    
    # @partial(jax.jit, static_argnums=(0, 2, 3), static_argnames=("model",))
    def solve_reverse_bridge(self, rng_key: jax.Array, xT: jnp.ndarray, n_batches: int,model = None) -> SamplePath:
        
        reverse_bridge = self.sde.get_reverse_bridge(model)
        reverse_bridge_solver = EulerMaruyama(reverse_bridge, self.wiener)
        sol = reverse_bridge_solver.solve(rng_key, x0=xT, n_batches=n_batches)

        return sol
    
    def dsm_loss(self, outputs: jnp.ndarray, bs: jnp.ndarray):
        loss = jnp.sum(jnp.linalg.norm(outputs - bs, axis=-1)**2, axis=1) * self.dt
        loss = 0.5 * jnp.mean(loss, axis=0)
        return loss
    


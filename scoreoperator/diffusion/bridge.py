from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr

from scoreoperator.diffusion.sde import SDEFactory

class DiffusionBridge:
    """
    DiffusionBridge: A class for handling forward and reverse processes in diffusion bridges.
    This class provides methods for solving forward unconditioned SDEs (for training),
    reverse bridge SDEs (for inference), and computing DSM losses.
    Attributes:
        sde: An SDE instance created by SDEFactory based on the specified type and config.
    Methods:
        __init__(sde_type, sde_config): Initialize with specified SDE type and configuration.
        solve_forward_sde(rng_key, x0): Solve the forward SDE process starting from initial point x0.
        solve_reverse_bridge(rng_key, xT, model): Solve the reverse bridge process starting from terminal point xT.
        dsm_loss(preds, bs): Calculate denoising score matching loss between predictions and targets.
    """

    def __init__(self, sde_type, sde_config):
        self.sde = SDEFactory.create(sde_type, sde_config)

    @partial(jax.jit, static_argnums=(0,))
    def solve_forward_sde(self, rng_key, x0):
        dWs = jr.normal(rng_key, shape=(self.sde.n_steps,) + self.sde.bm_shape) * jnp.sqrt(self.sde.dt)
        
        def scan_body(carry, val):
            x, t = carry
            dt, dW = val
            drift = self.sde.f(t, x) * dt
            Phi = self.sde.g(t, x)
            diffusion = self.sde.apply_g(Phi, dW)
            x_next = x + drift + diffusion
            t_next = t + dt
            b = - diffusion / dt
            return (x_next, t_next), (x_next, t_next, b)
            
        *_, (xs, ts, bs) = jax.lax.scan(
            scan_body,
            init=(x0, jnp.array(0.0)),
            xs=(self.sde.dts, dWs),
            length=self.sde.n_steps
        )

        return xs, ts, bs        # x, s, b_s(x; x(t_{n-1}))
    
    def solve_reverse_bridge(self, rng_key, x0, xT, model):
        dWs = jr.normal(rng_key, shape=(self.sde.n_steps,) + self.sde.bm_shape) * jnp.sqrt(self.sde.dt)
        self.sde.x0 = x0
        self.sde.xT = xT
        reversed_sde = self.sde.get_reverse_bridge(self.sde, model)
        
        def scan_body(carry, val):
            y, t = carry
            dt, dW = val
            drift = reversed_sde.f(t, y) * dt
            Phi = reversed_sde.g(t, y)
            diffusion = reversed_sde.apply_g(Phi, dW)
            y_next = y + drift + diffusion
            tau_next = t + dt
            return (y_next, tau_next), (y_next)
            
        *_, ys = jax.lax.scan(
            scan_body,
            init=(xT, jnp.array(0.0)),
            xs=(self.sde.dts, dWs),
            length=self.sde.n_steps,
        )
        
        return ys
    
    def dsm_loss(self, preds, bs):
        b, t, *_, d = preds.shape
        loss = (preds - bs).reshape(b, t, -1, d)
        loss = jnp.mean(jnp.sum(jnp.square(loss), axis=-1), axis=-1) 
        loss = jnp.sum(loss, axis=1) * self.sde.dt
        loss = 0.5 * jnp.mean(loss, axis=0)
        return loss
    


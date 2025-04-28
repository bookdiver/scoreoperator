from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr

from scoreoperator.diffusion.sde import BaseSDE, ProjectedSDE, SDEFactory

class DiffusionBridge:
    sde: BaseSDE
    projected_sde: ProjectedSDE

    def __init__(self, sde_type, sde_config):
        self.sde = SDEFactory.create(sde_type, sde_config)
        self.projected_sde = None
        
    def project(self, x_shape, t_shape, w_shape):
        self.projected_sde = self.sde.project(x_shape, t_shape, w_shape)

    @partial(jax.jit, static_argnums=(0,))
    def solve_forward_sde(self, rng_key):
        assert self.projected_sde is not None, "Infinite-dimensional SDE must be projected to finite-dimensional space by .project()"
        
        n_steps = self.projected_sde.t_shape[0]
        w_shape = self.projected_sde.w_shape
        dt = self.projected_sde.dt
        x0_ = self.projected_sde.x0_
        
        dWs = jr.normal(rng_key, shape=(n_steps,) + w_shape) * jnp.sqrt(dt)
        
        def scan_body(carry, val):
            x, t = carry
            dW = val
            drift = self.projected_sde.f(t, x) * dt
            Phi = self.projected_sde.g(t, x)
            diffusion = self.projected_sde.apply_g(Phi, dW)
            x_next = x + drift + diffusion
            t_next = t + dt
            b_ = - diffusion / dt
            return (x_next, t_next), (x_next, t_next, b_)
            
        *_, (xs, ts, bs) = jax.lax.scan(
            scan_body,
            init=(x0_, jnp.array(0.0)),
            xs=dWs,
            length=n_steps
        )
        
        xs = jnp.concatenate([x0_[None, ...], xs], axis=0)
        ts = jnp.concatenate([jnp.array([0.0]), ts], axis=0)
        return xs[:-1], ts[:-1], bs     
    
    def solve_reverse_bridge(self, rng_key, score_model, x_shape, t_shape, w_shape, xT = None):
        reversed_projected_bridge = self.sde.get_reverse_projected_bridge(
            self.sde, 
            score_model, 
            x_shape, 
            t_shape, 
            w_shape, 
            xT
        )
        
        n_steps = t_shape[0]
        dt = reversed_projected_bridge.dt
        
        dWs = jr.normal(rng_key, shape=(n_steps,) + w_shape) * jnp.sqrt(dt)
        
        def scan_body(carry, val):
            y, t = carry
            dW = val
            drift = reversed_projected_bridge.f(t, y) * dt
            Phi = reversed_projected_bridge.g(t, y)
            diffusion = reversed_projected_bridge.apply_g(Phi, dW)
            y_next = y + drift + diffusion
            t_next = t + dt
            return (y_next, t_next), y_next
            
        *_, ys = jax.lax.scan(
            scan_body,
            init=(reversed_projected_bridge.x0_, jnp.array(0.0)),
            xs=dWs,
            length=n_steps,
        )
        
        ys = jnp.concatenate([reversed_projected_bridge.x0_[None, ...], ys], axis=0)
        return ys
    
    def dsm_loss(self, preds, bs):
        assert self.projected_sde is not None, "Infinite-dimensional SDE must be projected to finite-dimensional space by .project()"
        b, t, *_, d = preds.shape
        loss = (preds - bs).reshape(b, t, -1, d)
        loss = jnp.mean(jnp.sum(jnp.square(loss), axis=-1), axis=-1) 
        loss = jnp.sum(loss, axis=1) * self.projected_sde.dt
        loss = 0.5 * jnp.mean(loss, axis=0)
        return loss
    


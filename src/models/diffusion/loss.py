import jax
import jax.numpy as jnp

def weighted_square_norm(x: jnp.ndarray, w: jnp.ndarray):
    return x.T @ w @ x

def dsm_loss(predss: jnp.ndarray, gradss: jnp.ndarray, covss: jnp.ndarray, dt: float):
    # preds: (b_size, t_size, d_size), grads: (b_size, t_size, d_size), covs: (b_size, t_size, d_size, d_size)
    losses = jax.vmap(jax.vmap(weighted_square_norm))(predss+gradss, covss) * dt   # (b_size, t_size)
    losses = jnp.sum(losses, axis=1)    # (b_size,)
    loss = jnp.mean(losses)   # ()
    return loss
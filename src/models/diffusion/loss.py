import jax.numpy as jnp

def dsm_loss(predss: jnp.ndarray, gradss: jnp.ndarray, dt: float):
    losses = jnp.einsum("bti,bti->bt", predss+gradss, predss+gradss) # (b_size, t_size)
    losses = jnp.sum(losses, axis=1)    # (b_size,
    loss = 0.5 * dt * jnp.mean(losses)   # ()
    return loss
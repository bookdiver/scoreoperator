import jax
import jax.numpy as jnp

class DiffusionLoader:
    def __init__(self, sde, seed, init_cond, batch_size, shuffle=True):
        self._sde = sde
        self._key = jax.random.PRNGKey(seed)
        self._init_cond = init_cond
        self._batch_size = batch_size
        self._shuffle = shuffle

    def __iter__(self):
        return self
    
    def __next__(self):
        key, self._key = jax.random.split(self._key)
        ts = self._sample_t()
        keys = jax.random.split(key, self._batch_size)
        xs, ts, zs = jax.vmap(self._sde.sample, in_axes=(0, 0, None, None))(keys, ts, self._init_cond, True)
        return xs, ts, zs
    
    def _sample_t(self):
        if self._shuffle:
            ts = jax.random.uniform(self._key, (self._batch_size, ))
        else:
            ts = jnp.linspace(0, 1, self._batch_size)
        return ts
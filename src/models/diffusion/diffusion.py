import jax
import jax.numpy as jnp

class DiffusionLoader:
    def __init__(self, sde, seed, n_steps, init_cond, batch_size):
        self._sde = sde
        self._key = jax.random.PRNGKey(seed)
        self._init_cond = init_cond
        self._batch_size = batch_size
        self._n_steps = n_steps

    @property
    def ts(self):
        return jnp.linspace(0., 1., self._n_steps+1)[1:]

    def __iter__(self):
        return self
    
    def __next__(self):
        key, self._key = jax.random.split(self._key)
        ts = self._sample_t()
        keys = jax.random.split(key, self._batch_size)
        xs, ts, zs, sigmas = jax.vmap(self._sde.sample, in_axes=(0, 0, None, None))(keys, ts, self._init_cond, True)
        return xs, ts, zs, sigmas
    
    def _sample_t(self):
        t_idx = jax.random.randint(self._key, (self._batch_size, ), 0, self._n_steps)
        ts = self.ts[t_idx]
        return ts
import abc

import jax.numpy as jnp
from jax.random import PRNGKey

class Function(abc.ABC):
    do_dim: int
    co_dim: int
    eps: float = 1e-2

    @abc.abstractmethod
    def noise(self, n_samples: int, rng_key: PRNGKey):
        pass

    @abc.abstractmethod
    def _sample(self, n_samples: int) -> jnp.ndarray:
        pass

    def sample(self, n_samples: int, rng_key: PRNGKey = None) -> jnp.ndarray:
        if rng_key is None:
            return self._sample(n_samples)
        else:
            noise = self.noise(rng_key)
            return self._sample(n_samples) + noise
    
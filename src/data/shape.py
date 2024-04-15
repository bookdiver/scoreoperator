import abc

import jax.numpy as jnp

class Shape(abc.ABC):

    @abc.abstractmethod
    def sample(self, n_samples: int) -> jnp.ndarray:
        pass
    
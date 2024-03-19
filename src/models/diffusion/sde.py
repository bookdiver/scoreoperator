import abc

import jax
import jax.numpy as jnp

class SDE(abc.ABC):
    def __init__(self, gp):
        self._gp = gp

    @abc.abstractmethod
    def f(self, t, x, *args):
        pass

    @abc.abstractmethod
    def g(self, t, x, *args):
        pass

    @abc.abstractmethod
    def _transition_prob(self, t, x, x0):
        pass

    def sample(self, key, t, x0, verbose=False):
        mean, std = self._transition_prob(t, x0)
        z = self._gp.sample(key)
        xt = mean + std * z
        if verbose:
            return (xt, t, z)
        else:
            return xt
        
    def sample_batch(self, key, t, x0, num_batches, verbose):
        keys = jax.random.split(key, num_batches)
        return jax.vmap(self.sample, in_axes=(0, None, None, None))(keys, t, x0, verbose)
    

class BrownianSDE(SDE):
    def __init__(self, gp, sigma):
        super().__init__(gp)
        self._sigma = sigma

    def f(self, t, x):
        return jnp.zeros_like(x)
    
    def g(self, t, x):
        return self._sigma * jnp.eye(x.shape[-1])
    
    def _transition_prob(self, t, x, x0):
        mean = x0
        std = jnp.sqrt(self._sigma * t) 
        return mean, std


class VPSDE(SDE):
    def __init__(self, gp, betas):
        super().__init__(gp)
        self._beta_min, self._beta_max = betas

    def beta(self, t):
        return self._beta_min + (self._beta_max - self._beta_min) * t
    
    def f(self, t, x):
        return -0.5 * self.beta(t) * x
    
    def g(self, t, x):
        return jnp.sqrt(self.beta(t)) * jnp.eye(x.shape[-1])

    def transition_prob(self, t, x0):
        log_mean_coeff = -0.25 * t**2 * (self._beta_max - self._beta_min) - 0.5 * t * self._beta_min
        mean = jnp.exp(log_mean_coeff) * x0
        std = jnp.sqrt(1 - jnp.exp(2.0 * log_mean_coeff))
        return mean, std
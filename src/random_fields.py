import abc
import jax
import jax.numpy as jnp
from jax.scipy.fft import idctn

class RandomField2D(abc.ABC):
    def __init__(self, size):
        self.size = size
    
    @abc.abstractmethod
    def sample(self, key, n_samples):
        pass

class IndependentGRF2D(RandomField2D):
    """ Independent Gaussian random field, i.e. white noise field """
    def __init__(self, size, sigma):
        super().__init__(size)
        self.sigma = sigma
    
    def sample(self, key, n_samples):
        z = self.sigma * jax.random.normal(key, (n_samples, *self.size, 2))
        return z
    
class GRF2D(RandomField2D):
    """ Gaussian random field with covariance operator as C = sigma * (-Delta + tau^2)^{-alpha}"""
    def __init__(self, size, sigma, alpha, tau):
        super().__init__(size)
        self.sigma = sigma
        k1, k2 = jnp.arange(self.size[0]), jnp.arange(self.size[1])
        K1, K2 = jnp.meshgrid(k1, k2)
        C = (jnp.pi**2) * (K1**2 + K2**2) + tau**2
        C = jnp.power(C, -alpha/2.0)
        C = jnp.power(tau, alpha-1) * C
        self.coeff = C
    
    def sample(self, key, n_samples):
        xr = jax.random.normal(key, (n_samples, *self.size, 2))
        xr = jnp.einsum('ij,nijk->nijk', self.coeff, xr)
        xr = jnp.sqrt(self.size[0]*self.size[1]) * xr
        xr = xr.at[:, 0, 0].set(0.0)
        z = idctn(xr, axes=(1, 2))
        return z
    
class PeriodicGRF2D(RandomField2D):
    """ Gaussian random field with periodic boundary conditions """
    def __init__(self, size, sigma, alpha, tau):
        super().__init__(size)
        self.sigma = sigma if sigma else jnp.power(tau, 0.5*(2*alpha-2.0))

        freqs1 = jnp.concatenate(
            [jnp.arange(0, self.size[0]//2),
             jnp.arange(-self.size[0]//2, 0)],
            axis=0
        )
        k1 = jnp.tile(freqs1.reshape(-1, 1), reps=(1, self.size[1]))
        freqs2 = jnp.concatenate(
            [jnp.arange(0, self.size[1]//2),
             jnp.arange(-self.size[1]//2, 0)],
            axis=0
        )
        k2 = jnp.tile(freqs2.reshape(1, -1), reps=(self.size[0], 1))
        sqrt_eig = (
            self.size[0] * self.size[1] * sigma * jnp.sqrt(2.0) * jnp.power(
                (4.0 * jnp.pi**2 * (k1**2 + k2**2) + tau**2), -alpha/2.0
            )   
        )
        sqrt_eig = sqrt_eig.at[0, 0].set(0.0)
        sqrt_eig = sqrt_eig.at[jnp.logical_and(k1+k2<=0, jnp.logical_or(k1+k2!=0.0, k1<=0.0))].set(0.0)
        self.sqrt_eig = sqrt_eig

    def sample(self, key, n_samples):
        x = jax.random.normal(key, (n_samples, *self.size, 2), dtype=jnp.complex64)
        x = jnp.einsum('ij,nijk->nijk', self.sqrt_eig, x)
        z = jnp.fft.ifft2(x, s=self.size, axes=(1, 2)).real
        return z




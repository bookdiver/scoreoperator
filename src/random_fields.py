import abc
import jax
import jax.numpy as jnp
import jax.scipy as jsp

class RandomField(abc.ABC):
    def __init__(self, dim, size, **kwargs):
        self._dim = dim
        self._size = size
    
    @property
    def _sizes(self):
        return (self._size for _ in range(self._dim))

    @abc.abstractmethod
    def sample(self, n_samples, key):
        pass

class IndependentGRF(RandomField):
    """ Independent Gaussian random field, i.e. white noise field """
    def __init__(self, dim, size, sigma, **kwargs):
        super().__init__(dim, size)
        self._sigma = sigma
    
    def sample(self, n_samples, key):
        z = self._sigma * jax.random.normal(key, (n_samples, *self._sizes))
        return z
    
class GRF(RandomField):
    """ Gaussian random field with covariance operator as C = sigma * (-Delta + tau^2)^{-alpha}"""
    def __init__(self, dim, size, sigma, alpha, tau, bc, **kwargs):
        super().__init__(dim, size)
        self._sigma = sigma if sigma else jnp.power(tau, 0.5*(2*alpha-2.0))
        self._bc = bc.lower()
        if self._bc == 'none':
            freqs = jnp.arange(self._size)
        elif self._bc == 'periodic':
            freqs = jnp.concatenate(
                [jnp.arange(0, self._size//2),
                jnp.arange(-self._size//2, 0)],
                axis=0
            )
        else:
            raise ValueError(f"Unknown boundary condition: {bc}, can only be 'none' or 'periodic'")
        
        if dim == 1:
            sqrt_eig = self._size * jnp.sqrt(2.0) * self._sigma * \
                jnp.power((jnp.pi**2 * freqs**2 + tau**2), -alpha/2.0)
            self._sqrt_eig = sqrt_eig.at[0].set(0.0)
        elif dim == 2:
            k1 = jnp.tile(freqs.reshape(-1, 1), reps=(1, self._size))
            k2 = jnp.transpose(k1, axes=(1, 0))
            sqrt_eig = self._size**2 * jnp.sqrt(2.0) * self._sigma * \
                jnp.power((jnp.pi**2 * (k1**2 + k2**2) + tau**2), -alpha/2.0)
            self._sqrt_eig = sqrt_eig.at[0, 0].set(0.0)
        else:
            raise ValueError(f"{dim}-dimenisonal GRF has not been implemented")
    
    def sample(self, n_samples, key):
        if self._bc == 'none':
            x = jax.random.normal(key, (n_samples, *self._sizes), dtype=jnp.float32)
            if self._dim == 1:
                x_dct = jsp.fft.dct(x, n=self._size, axis=1)
                x = jnp.einsum('i,ni->ni', self._sqrt_eig, x_dct)
                z = jsp.fft.idct(x, n=self._size, axis=1)
            elif self._dim == 2:
                x_dct = jsp.fft.dctn(x, s=self._sizes, axes=(1, 2))
                x = jnp.einsum('ij,nij->nij', self._sqrt_eig, x)
                z = jsp.fft.idctn(x, s=self._sizes, axes=(1, 2))
        elif self._bc == 'periodic':
            x = jax.random.normal(key, (n_samples, *self._sizes), dtype=jnp.complex64)
            if self._dim == 1:
                x_dft = jnp.fft.fft(x, n=self._size, axis=1)
                x = jnp.einsum('i,ni->ni', self._sqrt_eig, x_dft)
                z = jnp.fft.ifft(x, n=self._size, axis=1).real
            elif self._dim == 2:
                x_dft = jnp.fft.fft2(x, s=self._sizes, axes=(1, 2))
                x = jnp.einsum('ij,nij->nij', self._sqrt_eig, x_dft)
                z = jnp.fft.ifft2(x, s=self._sizes, axes=(1, 2)).real
        return z
    
class PeriodicGRF(RandomField):
    """ Gaussian random field with periodic boundary conditions """
    def __init__(self, dim, size, sigma, alpha, tau, **kwargs):
        super().__init__(dim, size)
        self._sigma = sigma if sigma else jnp.power(tau, 0.5*(2*alpha-2.0))

        if dim == 1:
            freqs = jnp.concatenate(
                [jnp.arange(0, self._size//2),
                jnp.arange(-self._size//2, 0)],
                axis=0
            )
            sqrt_eig = self._size * jnp.sqrt(2.0) * self._sigma * \
                jnp.power((4.0 * jnp.pi**2 * freqs**2 + tau**2), -alpha/2.0)
            self._sqrt_eig = sqrt_eig.at[0].set(0.0)
        elif dim == 2:
            freqs = jnp.concatenate(
                [jnp.arange(0, self._size//2),
                jnp.arange(-self._size//2, 0)],
                axis=0
            )
            k1 = jnp.tile(freqs.reshape(-1, 1), reps=(1, self._size))
            k2 = jnp.transpose(k1, axes=(1, 0))
            sqrt_eig = self._size**2 * jnp.sqrt(2.0) * self._sigma * \
                jnp.power((4.0 * jnp.pi**2 * (k1**2 + k2**2) + tau**2), -alpha/2.0)
            self._sqrt_eig = sqrt_eig.at[0, 0].set(0.0)
        else:
            raise ValueError(f"{dim}-dimenisonal Periodic GRF has not been implemented")

    def sample(self, n_samples, key):
        xi = jax.random.normal(key, (n_samples, *self._sizes), dtype=jnp.complex64)
        if self._dim == 1:
            xi = jnp.einsum('i,ni->ni', self._sqrt_eig, xi)
            z = jnp.fft.fft(xi, n=self._size).imag
        elif self._dim == 2:
            xi = jnp.einsum('ij,nij->nij', self._sqrt_eig, xi)
            z = jnp.fft.ifft2(xi, s=self._sizes, axes=(1, 2)).imag
        return z


class GaussianRandomField:
    def __init__(self, dim, size, gf_type, **kwargs):
        assert gf_type.lower() in ['igrf', 'grf', 'pgrf'], f"Unknown random field type: {gf_type}, can only be 'wn', 'grf', or 'pgrf'"
        if gf_type.lower() == 'igrf':
            self._gf = IndependentGRF(dim, size, **kwargs)
        elif gf_type.lower() == 'grf':
            self._gf = GRF(dim, size, bc='none', **kwargs)
        elif gf_type.lower() == 'pgrf':
            self._gf = GRF(dim, size, bc='periodic', **kwargs)
        
    def sample(self, n_samples, key):
        return self._gf.sample(n_samples, key)


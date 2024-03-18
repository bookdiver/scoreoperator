from functools import partial
import jax
import jax.numpy as jnp

def gaussian_kernel(rho, sigma, kappa):
    return sigma**2 * jnp.exp(-0.5 * (rho / kappa)**2)

def matern_kernel(rho, sigma, kappa, nu):
    if nu == 0.5:
        return sigma**2 * jnp.exp(-rho / kappa)
    elif nu == 1.5:
        return sigma**2 * (1.0 + jnp.sqrt(3.0) * rho / kappa) * jnp.exp(-jnp.sqrt(3.0) * rho / kappa)
    elif nu == 2.5:
        return sigma**2 * (1.0 + jnp.sqrt(5.0) * rho / kappa + 5.0 * rho**2 / (3.0 * kappa**2)) * jnp.exp(-jnp.sqrt(5.0) * rho / kappa)
    else:
        raise ValueError(f"nu={nu} has not been implemented")
    
def delta_kernel(rho, sigma):
    return sigma**2 * jnp.eye(rho.shape[-1])

class GaussianProcess:
    def __init__(self, input_dim, output_dim, n_sample_pts, kernel_type, **kwargs):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._n_sample_pts = n_sample_pts

        if kernel_type == 'delta':
            _kernel_cls = partial(delta_kernel, **kwargs)
        elif kernel_type == 'gaussian':
            _kernel_cls = partial(gaussian_kernel, **kwargs)
        elif kernel_type == 'matern':
            _kernel_cls = partial(matern_kernel, **kwargs)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}, can only be 'delta', 'gaussian', or 'matern'")
        
        self._grid = self._get_grid()
        self._L = self._get_L(_kernel_cls, self._grid)

    @property
    def _input_size(self):
        return (self._n_sample_pts, ) * self._input_dim
    
    @property
    def _n_total_pts(self):
        return self._n_sample_pts ** self._input_dim

    def _get_L(self, kernel, grid, eps=1e-4):
        C = kernel(self.cdist(grid, grid))
        return jnp.linalg.cholesky(C + eps * jnp.eye(C.shape[-1]))

    def sample(self, key):
        x = jax.random.normal(key, (self._n_total_pts, self._output_dim))
        z = jnp.einsum('ij,jd->id', self._L, x)
        z = z.reshape(*self._input_size, self._output_dim)
        return z
    
    def sample_batch(self, key, n_samples):
        keys = jax.random.split(key, n_samples)
        return jax.vmap(self.sample)(keys)
    
    def _get_grid(self):
        if self._input_dim == 1:
            return jnp.linspace(0, 1, self._n_sample_pts+1)[:-1]
        elif self._input_dim == 2:
            xs = jnp.linspace(0, 1, self._n_sample_pts+1)[:-1]
            ys = jnp.linspace(0, 1, self._n_sample_pts+1)[:-1]
            xx, yy = jnp.meshgrid(xs, ys, indexing='xy')
            return jnp.concatenate(
                [xx.reshape(-1, 1), yy.reshape(-1, 1)],
                axis=-1
            )
        else:
            raise NotImplementedError
    
    @staticmethod
    def cdist(grid1, grid2):
        assert grid1.ndim == grid2.ndim
        if grid1.ndim == 1:
            grid1 = jnp.expand_dims(grid1, axis=-1)
            grid2 = jnp.expand_dims(grid2, axis=-1)
        grid1_grid2 = grid1[:, None, :] - grid2[None, :, :]
        return jnp.sqrt(jnp.sum(jnp.square(grid1_grid2), axis=-1))
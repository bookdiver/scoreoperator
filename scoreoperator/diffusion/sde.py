import abc
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp

class BaseSDE(abc.ABC):
    T: float
    dt: float
    n_steps: int
    ts: jnp.ndarray
    dts: jnp.ndarray
    x0: jnp.ndarray
    xT: jnp.ndarray
    bm_shape: tuple[int]

    def __init__(self, config):
        super().__init__()
        self.T = config['T']
        self.dt = config['dt']
        self.x0 = config['x0']
        self.xT = config['xT']
        self.bm_shape = config['bm_shape']
        
    @property
    def ts(self):
        return jnp.arange(0.0, self.T + self.dt, self.dt)
    
    @property
    def n_steps(self):
        return len(self.ts) - 1
    
    @property
    def dts(self):
        return jnp.diff(self.ts)
        
    @abc.abstractmethod
    def f(self, t, x):
        pass

    @abc.abstractmethod
    def g(self, t, x):
        pass
    
    @abc.abstractmethod
    def apply_g(self, Phi, dW):
        """ Apply the Hilbert-Schmidt operator Phi=g(X(t)) on the Wiener process dW
        """
        pass

    @staticmethod
    def get_reverse_bridge(forward_sde, score):
        """ Get the reverse bridge SDE from the additional drift model.

        Args:
            model (Model): a wrapped model class that acts as the approximation of the additional drift.

        Raises:
            ValueError: Unknown model matching object

        Returns:
            BaseSDE: reverse bridge SDE class
        """
        reversed_sde_config = {
            'T': forward_sde.T,
            'dt': forward_sde.dt,
            'x0': forward_sde.xT,
            'xT': forward_sde.x0,
            'bm_shape': forward_sde.bm_shape
        }
        
        forward_f = forward_sde.f
        forward_g = forward_sde.g
        forward_apply_g = forward_sde.apply_g
        
        class ReverseSDE(BaseSDE):
            def __init__(self):
                super().__init__(reversed_sde_config)

            def f(self, t, y):
                tau = self.T - t
                return -forward_f(tau, y) + score(tau, y)
            
            def g(self, t, y):
                tau = self.T - t
                return forward_g(tau, y)
            
            def apply_g(self, Phi, dW):
                return forward_apply_g(Phi, dW)
        
        return ReverseSDE()
    
class BrownianSDE(BaseSDE):
    """ Brownian motion SDE: dX(t) = sigma * dW(t)
    """
    sigma: float
    
    def __init__(self, config):
        super().__init__(config)
        self.sigma = config["sigma"]

    def f(self, t, x):
        return jnp.zeros_like(x)
    
    def g(self, t, x):
        return None
    
    def apply_g(self, Phi, dW):
        return self.sigma * dW

    

# class LagrangianSDE(BaseSDE):
#     """ Lagrangian SDE: dX(t) = Q^{1/2}(X(t)) dW(t) with noise fields assigned to each landmark, 
#         see ``Stochastic flows and shape bridges, S. Sommer et al.'' for details.
#     """
#     def __init__(self, 
#                  sigma: float = 1.0, 
#                  kappa: float = 0.1,
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.sigma = sigma
#         self.kappa = kappa
    
#     def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
#         return jnp.zeros_like(x)
    
#     def g(self, t: float, x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
#         """ Diffusion term of the Lagrangian SDE defined by the Gaussian kernel k(x, y) = sigma * exp(-||x-y||^2 / kappa^2).
#             The covariance is computed between the landmarks.

#         Args:
#             t (float): time step.
#             x (jnp.ndarray): flatten function evaluation x, shape (n_pts*co_dim, ).
#             eps (float, optional): regularization to avoid singularity of the diffusion term. Defaults to 1e-4.

#         Returns:
#             jnp.ndarray: diffusion term, shape (n_pts*co_dim, n_pts*co_dim).
#         """
#         x = x.reshape(-1, 2)
#         n_pts = x.shape[0]
#         kernel_fn = lambda x: self.sigma * jnp.exp(-jnp.linalg.norm(x, axis=-1)**2 / self.kappa**2)
#         dist = x[:, None, :] - x[None, :, :]
#         kernel = kernel_fn(dist) + eps * jnp.eye(n_pts)     # Regularization to avoid singularity
#         Q_half = jnp.einsum("ij,kl->ikjl", kernel, jnp.eye(2))
#         Q_half = Q_half.reshape(2*n_pts, 2*n_pts)
#         return Q_half

class EulerianSDE(BaseSDE):
    """ 
    Eulerian SDE: dX(t) = Q^{1/2}(X(t)) dW(t) with noise fields acting on the whole domain, 
    see ``Stochastic flows and shape bridges, S. Sommer et al.'' for details.
    """
    k_alpha: float
    k_sigma: float
    
    def __init__(self, config):
        super().__init__(config)
        self.k_alpha = config["k_alpha"]
        self.k_sigma = config["k_sigma"]
    
    def f(self, t, x):
        return jnp.zeros_like(x)
    
    def g(self, t, x):
        return x + self.x0 # NOTE: it is not formally true, as we will use x to compute Phi dW

    @partial(jax.jit, static_argnums=(0,))
    def apply_g(self, Phi, dW):
        window_size = 17
        window_span = (-2.0, 2.0)
        scaling = jnp.abs(self.bm_shape[0] / (window_span[0] - window_span[1]))
        center = jnp.array([self.bm_shape[0] / 2, self.bm_shape[1] / 2])
        
        coords_to_pixels = lambda x: scaling * x + center[jnp.newaxis, :]
        
        delta_x = (window_span[1] - window_span[0]) / (window_size - 1)
        window_xs = jnp.linspace(*window_span, window_size)
        window_scale = self.k_sigma / delta_x
        
        def convolution_window(span, scale):
            window = jsp.stats.norm.pdf(span, 0, scale) \
                    * jsp.stats.norm.pdf(span[:, None], 0, scale)
            window /= jnp.sqrt(jnp.sum(window**2, axis=(0, 1)))
            return window
        
        normalized_window = convolution_window(window_xs, window_scale)
        window = self.k_alpha * normalized_window
        dW_convolved = jax.vmap(
            partial(jsp.signal.convolve, mode="same"),
            in_axes=(2, None),
            out_axes=2
        )(dW, window)
        
        x_pixels = coords_to_pixels(Phi)  # shape (N, 2)
        x_coords = x_pixels.T  # shape (2, N)

        def interp_channel(channel):  # channel: shape (H, W)
            return jsp.ndimage.map_coordinates(channel, x_coords, order=1, mode="nearest")

        Phi_dW = jax.vmap(interp_channel, in_axes=2, out_axes=1)(dW_convolved)  # shape (C, N)
        return Phi_dW
            

class SDEFactory:
    """
    Factory class for creating different SDE instances.
    """
    @staticmethod
    def create(sde_type, sde_config):
        sde_classes = {
            "brownian": BrownianSDE,
            "eulerian": EulerianSDE,
        }

        if sde_type.lower() not in sde_classes:
            raise ValueError(f"Unknown SDE type: {sde_type}")

        sde_class = sde_classes[sde_type.lower()]
        return sde_class(sde_config)
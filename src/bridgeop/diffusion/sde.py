from __future__ import annotations
import abc
from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp

# from ...data.function import FunctionData
# from .model import Model

class BaseSDE(abc.ABC):
    W_shape: Tuple[int, ...]
    T: float

    def __init__(self, T: float = 1.0, **kwargs):
        super().__init__()
        self.T = T
        self.W_shape = kwargs.get("W_shape", None)
        
    @abc.abstractmethod
    def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        """ Drift term of the SDE.

        Args:
            t (float): time step
            x (jnp.ndarray): flatten function evaluation x, shape (n_pts*co_dim, )

        Returns:
            jnp.ndarray: drift term, shape (n_pts*co_dim, )
        """
        pass

    @abc.abstractmethod
    def g(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        """ Diffusion term of the SDE.

        Args:
            t (float): time step
            x (jnp.ndarray): flatten function evaluation x, shape (n_pts*co_dim, )

        Returns:
            jnp.ndarray: diffusion term, shape (n_pts*co_dim, noise_dim)
        """
        pass

    def a(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        """ Covariance term of the SDE.

        Args:
            t (float): time step
            x (jnp.ndarray): flatten function evaluation x, shape (n_pts*co_dim, )

        Returns:
            jnp.ndarray: covariance term, shape (n_pts*co_dim, n_pts*co_dim)
        """
        g = self.g(t, x)
        return jnp.dot(g, g.T)
    
    def get_reverse_bridge(self, model) -> BaseSDE:
        """ Get the reverse bridge SDE from the additional drift model.

        Args:
            model (Model): a wrapped model class that acts as the approximation of the additional drift.

        Raises:
            ValueError: Unknown model matching object

        Returns:
            BaseSDE: reverse bridge SDE class
        """
        T = self.T
        W_shape = self.W_shape
        
        f = self.f
        g = self.g
        a = self.a

        drift_fn = lambda t, x: model(t, x)

        class ReverseSDE(BaseSDE):
            def __init__(self):
                super().__init__(T=T, **({"W_shape": W_shape} if W_shape is not None else {}))

            def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
                reversed_t = self.T - t
                # return -f(t=reversed_t, x=x) + drift_fn(t=reversed_t, x=x) + self.jac_a(t=reversed_t, x=x)
                return -f(t=reversed_t, x=x) + drift_fn(t=reversed_t, x=x)
            
            def g(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
                reversed_t = self.T - t
                return g(t=reversed_t, x=x)
            
            def a(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
                reversed_t = self.T - t
                return a(t=reversed_t, x=x)
            
            def jac_a(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
                reversed_t = self.T - t
                jacobian_ = jax.jacfwd(a, argnums=1)(reversed_t, x)
                return jnp.trace(jacobian_)
        
        return ReverseSDE()
    
class BrownianSDE(BaseSDE):
    """ Brownian motion SDE: dX(t) = sigma * dW(t)
    """
    def __init__(self,
                 sigma: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma

    def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(x)
    
    def g(self, t: float, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self.sigma * jnp.eye(x.shape[-1])
    
    def a(self, t: float, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self.sigma**2 * jnp.eye(x.shape[-1])

    def inv_a(self, t: float, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return 1.0 / self.sigma**2 * jnp.eye(x.shape[-1])
    
class OUSDE(BaseSDE):
    """ Ornstein-Uhlenbeck SDE: dX(t) = -theta * X(t) dt + sigma * dW(t)
    """
    def __init__(self, 
                 sigma: float = 1.0, 
                 theta: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.theta = theta
    
    def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return -self.theta * x
    
    def g(self, t: float, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self.sigma * jnp.eye(x.shape[-1])
    
    def a(self, t: float, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self.sigma**2 * jnp.eye(x.shape[-1])
    
    def inv_a(self, t: float, x: jax.Array, **kwargs) -> jnp.ndarray:
        return 1.0 / self.sigma**2 * jnp.eye(x.shape[-1])

class LagrangianSDE(BaseSDE):
    """ Lagrangian SDE: dX(t) = Q^{1/2}(X(t)) dW(t) with noise fields assigned to each landmark, 
        see ``Stochastic flows and shape bridges, S. Sommer et al.'' for details.
    """
    def __init__(self, 
                 sigma: float = 1.0, 
                 kappa: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.kappa = kappa
    
    def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(x)
    
    def g(self, t: float, x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
        """ Diffusion term of the Lagrangian SDE defined by the Gaussian kernel k(x, y) = sigma * exp(-||x-y||^2 / kappa^2).
            The covariance is computed between the landmarks.

        Args:
            t (float): time step.
            x (jnp.ndarray): flatten function evaluation x, shape (n_pts*co_dim, ).
            eps (float, optional): regularization to avoid singularity of the diffusion term. Defaults to 1e-4.

        Returns:
            jnp.ndarray: diffusion term, shape (n_pts*co_dim, n_pts*co_dim).
        """
        x = x.reshape(-1, 2)
        n_pts = x.shape[0]
        kernel_fn = lambda x: self.sigma * jnp.exp(-jnp.linalg.norm(x, axis=-1)**2 / self.kappa**2)
        dist = x[:, None, :] - x[None, :, :]
        kernel = kernel_fn(dist) + eps * jnp.eye(n_pts)     # Regularization to avoid singularity
        Q_half = jnp.einsum("ij,kl->ikjl", kernel, jnp.eye(2))
        Q_half = Q_half.reshape(2*n_pts, 2*n_pts)
        return Q_half

class EulerianSDE(BaseSDE):
    """ Eulerian SDE: dX(t) = Q^{1/2}(X(t)) dW(t) with noise fields acting on the whole domain, 
        see ``Stochastic flows and shape bridges, S. Sommer et al.'' for details.
    """
    def __init__(self, 
                 sigma: float = 1.0, 
                 kappa: float = 0.1, 
                 W_shape: Tuple[int, ...] = (50, 50), 
                 W_range: Tuple[Tuple[float, float], ...] = ((-0.5, 1.5), (-0.5, 1.5)),
                 **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.kappa = kappa
        assert len(W_shape) == len(W_range), "W_shape and W_range must have the same length"
        self.W_shape = W_shape
        self.W_range = W_range
    
    @property
    def _W_grid(self):
        """ Noise grid.
        """
        grids = [jnp.linspace(start, end, num) for (start, end), num in zip(self.W_range, self.W_shape)]
        meshgrid = jnp.meshgrid(*grids, indexing='xy')
        grid = jnp.stack(meshgrid, axis=-1)
        return grid.reshape(-1, len(self.W_shape))
    
    def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(x)
    
    def g(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        """ Diffusion term of the Eulerian SDE defined by the Gaussian kernel k(x, y) = sigma * exp(-||x-y||^2 / kappa^2).
            The covariance is computed between the landmarks and the grid points.

        Args:
            t (float): time step.
            x (jnp.ndarray): flatten function evaluation x, shape (n_pts*co_dim, ).
            eps (float, optional): regularization to avoid singularity of the diffusion term. Defaults to 1e-4.

        Returns:
            jnp.ndarray: diffusion term, shape (n_pts*co_dim, noise_dim*co_dim).
        """
        x = x.reshape(-1, 2)
        n_pts = x.shape[0]
        kernel_fn = lambda x, y: self.sigma * jnp.exp(-0.5 * jnp.linalg.norm(x-y, axis=-1)**2 / self.kappa**2)
        Q_half = jax.vmap(
            jax.vmap(
                kernel_fn,
                in_axes=(None, 0),
                out_axes=0      # evaluate for all points of W_grid
            ),
            in_axes=(0, None),
            out_axes=0          # evaluate for all points of x
        )(x, self._W_grid)
        Q_half = jnp.einsum("ij,kl->ikjl", Q_half, jnp.eye(2))
        Q_half = Q_half.reshape(2*n_pts, 2*self._W_grid.shape[0])
        return Q_half
    
    def a(self, t: float, x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
        g = self.g(t, x)
        return jnp.dot(g, g.T) + eps * jnp.eye(g.shape[0])

class SDEFactory:
    """Factory class for creating different SDE instances."""

    @staticmethod
    def create(sde_type: str, **kwargs) -> BaseSDE:
        """
        Create an SDE instance based on the given type and parameters.

        Args:
            sde_type (str): The type of SDE to create.
            **kwargs: Additional parameters for the SDE initialization.

        Returns:
            BaseSDE: An instance of the specified SDE.

        Raises:
            ValueError: If an unknown SDE type is provided.
        """
        sde_classes = {
            "brownian": BrownianSDE,
            "ou": OUSDE,
            "lagrangian": LagrangianSDE,
            "eulerian": EulerianSDE,
        }

        if sde_type.lower() not in sde_classes:
            raise ValueError(f"Unknown SDE type: {sde_type}")

        sde_class = sde_classes[sde_type.lower()]
        return sde_class(**kwargs)

    @staticmethod
    def get_default_params(sde_type: str) -> Dict[str, Any]:
        """
        Get the default parameters for a given SDE type.

        Args:
            sde_type (str): The type of SDE.

        Returns:
            Dict[str, Any]: A dictionary of default parameters.

        Raises:
            ValueError: If an unknown SDE type is provided.
        """
        default_params = {
            "brownian": {"sigma": 1.0, "X0": None},
            "ou": {"sigma": 1.0, "theta": 1.0, "X0": None},
            "lagrangian": {"sigma": 1.0, "kappa": 0.1, "X0": None},
            "eulerian": {
                "sigma": 1.0,
                "kappa": 0.1,
                "W_shape": (50, 50),
                "W_range": ((-0.5, 1.5), (-0.5, 1.5))
            }
        }

        if sde_type.lower() not in default_params:
            raise ValueError(f"Unknown SDE type: {sde_type}")

        return default_params[sde_type.lower()]
import abc
import jax.numpy as jnp

class SDE(abc.ABC):
    dim: int

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        pass

    @abc.abstractmethod
    def g(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        pass

    def covariance(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        g = self.g(t, x)
        return jnp.dot(g, g.T)
    
    def inv_covariance(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.inv(self.covariance(t, x))
    
    def get_reverse_bridge(self, score_fn: callable = None) -> "SDE":
        dim = self.dim
        f = self.f
        g = self.g
        cov = self.covariance

        class ReverseSDE(SDE):
            def __init__(self):
                super().__init__()
                self.dim = dim

            def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
                return f(t, x) + jnp.dot(cov(t, x), score_fn(t, x))
            
            def g(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
                return g(t, x)
            
            def covariance(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
                return cov(t, x)
            
            def inv_covariance(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
                return jnp.linalg.inv(cov(t, x))
        
        return ReverseSDE()
    
class BrownianSDE(SDE):
    def __init__(self, dim: int, sigma: float = 1.0):
        super().__init__()
        self.dim = dim
        self.sigma = sigma

    def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(x)
    
    def g(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return self.sigma * jnp.eye(self.dim)
    
    def covariance(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return self.sigma**2 * jnp.eye(self.dim)

    def inv_covariance(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return 1.0 / self.sigma**2 * jnp.eye(self.dim)
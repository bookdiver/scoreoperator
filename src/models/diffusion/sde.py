import abc
import jax.numpy as jnp

class SDE(abc.ABC):

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
        f = self.f
        g = self.g
        cov = self.covariance

        class ReverseSDE(SDE):
            def __init__(self):
                super().__init__()

            def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
                reversed_t = 1.0 - t
                return f(t=reversed_t, x=x) + jnp.dot(cov(t=reversed_t, x=x), score_fn(t=reversed_t, x=x))
            
            def g(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
                reversed_t = 1.0 - t
                return g(t=reversed_t, x=x)
            
            def covariance(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
                reversed_t = 1.0 - t
                return cov(t=reversed_t, x=x)
            
            def inv_covariance(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
                reversed_t = 1.0 - t
                return jnp.linalg.inv(cov(t=reversed_t, x=x))
        
        return ReverseSDE()
    
class BrownianSDE(SDE):
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(x)
    
    def g(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return self.sigma * jnp.eye(x.shape[-1])
    
    def covariance(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return self.sigma**2 * jnp.eye(x.shape[-1])

    def inv_covariance(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return 1.0 / self.sigma**2 * jnp.eye(x.shape[-1])
import numpy as np
from scipy.interpolate import interp1d
import jax.numpy as jnp

from .function import Function

class Butterfly(Function):
    do_dim: int = 1
    co_dim: int = 2
    eps: float = 1e-2
    
    def __init__(self, name: str, interpolation: int = 512, interpolation_type: str = "linear"):
        self.pts = np.load("./src/data/raw/"+f"{name}.npy")

        ts = np.linspace(0, 1., len(self.pts))
        if interpolation and interpolation > len(self.pts):
            interp_ts = np.linspace(0, 1., interpolation)
            fx = interp1d(ts, self.pts[:, 0], kind=interpolation_type)
            fy = interp1d(ts, self.pts[:, 1], kind=interpolation_type)
            self.pts = jnp.array([fx(interp_ts), fy(interp_ts)]).T

    def noise(self, n_samples: int, rng_key):
        return jnp.zeros((n_samples, self.co_dim))

    def _sample(self, n_samples: int):
        if n_samples >= len(self.pts):
            return self.pts
        
        indices = jnp.linspace(0, len(self.pts)-1, n_samples, dtype=jnp.int32)
        return self.pts[indices]
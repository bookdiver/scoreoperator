import numpy as np
from scipy.interpolate import interp1d
import jax.numpy as jnp

from .shape import Shape

class Butterfly(Shape):
    def __init__(self, load_name: str, interpolation: int = 0, interpolation_type: str = "cubic"):
        self.pts = np.load("./src/data/raw/"+f"{load_name}.npy")

        ts = np.linspace(0, 1., len(self.pts))
        if interpolation and interpolation > len(self.pts):
            interp_ts = np.linspace(0, 1., interpolation)
            fx = interp1d(ts, self.pts[:, 0], kind=interpolation_type)
            fy = interp1d(ts, self.pts[:, 1], kind=interpolation_type)
            self.pts = jnp.array([fx(interp_ts), fy(interp_ts)]).T

    def sample(self, n_samples: int):
        if n_samples >= len(self.pts):
            return self.pts
        
        indices = jnp.linspace(0, len(self.pts)-1, n_samples, dtype=jnp.int32)
        return self.pts[indices]
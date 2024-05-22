import os
import numpy as np
from scipy.interpolate import interp1d
import jax.numpy as jnp

from .function import Function

class Butterfly(Function):
    do_dim: int = 1
    co_dim: int = 2
    eps: float = 1e-2
    
    def __init__(self, name: str, interp: int = None, interp_method: str = "linear"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, ".", "raw", f"{name}.npy")
        path = os.path.normpath(path)
        # self.pts = np.load("./src/data/raw/"+f"{name}.npy")
        self.pts = np.load(path)

        ts = np.linspace(0, 1., len(self.pts))
        if interp and interp > len(self.pts):
            interp_ts = np.linspace(0, 1., interp)
            fx = interp1d(ts, self.pts[:, 0], kind=interp_method)
            fy = interp1d(ts, self.pts[:, 1], kind=interp_method)
            self.pts = jnp.array([fx(interp_ts), fy(interp_ts)]).T

    def noise(self, n_samples: int, rng_key):
        return jnp.zeros((n_samples, self.co_dim))

    def _sample(self, n_samples: int):
        if n_samples >= len(self.pts):
            return self.pts
        
        indices = np.linspace(0, len(self.pts), n_samples, endpoint=False, dtype=np.int32)
        return self.pts[indices]
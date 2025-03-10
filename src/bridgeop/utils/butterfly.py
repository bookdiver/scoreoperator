import os
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d
import jax.numpy as jnp
from jax.random import PRNGKey, normal

from .function import FunctionData

class ButterflyData(FunctionData):
    """ Butterfly shape data, which is treated as a function R->R^2
    """
    do_dim: int = 1
    co_dim: int = 2
    eps: float = 1e-2           # variance of the Gaussian noise added on the data
    
    def __init__(self, specie_name: str, interp: int = None, interp_method: str = "linear"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, ".", "raw", f"{specie_name}.npy")
        path = os.path.normpath(path)
        if os.path.exists(path):                    # if there is a local file
            self.landmark_pts = np.load(path)
        else:
            raise NotImplementedError               # TODO: upload the raw data and access through links

        ts = np.linspace(0, 1., len(self.pts), endpoint=False)
        if interp and interp > len(self.pts):       # do the interpolation between read points
            interp_ts = np.linspace(0, 1., interp, endpoint=False)
            fx = interp1d(ts, self.pts[:, 0], kind=interp_method)
            fy = interp1d(ts, self.pts[:, 1], kind=interp_method)
            self.landmark_pts = jnp.array([fx(interp_ts), fy(interp_ts)]).T

    def _noise_fn(self, rng_key: PRNGKey) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """ Add Gaussian noise to the butterfly shape.
        """
        return lambda data: normal(rng_key, shape=data.shape) * jnp.sqrt(self.eps)
    
    def _add_noise(self, data: jnp.ndarray, rng_key: PRNGKey) -> jnp.ndarray:
        noise_fn = self._noise_fn(rng_key)
        return data + noise_fn(data)

    def _eval(self, n: int) -> jnp.ndarray:
        if n >= len(self.pts):      # take all of the points
            return self.pts
        else:                       # take a subset of the points
            indices = np.linspace(0, len(self.pts), n, endpoint=False, dtype=np.int32)
            return self.pts[indices]
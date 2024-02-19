import jax
import jax.numpy as jnp
from flax import linen as nn

class SpectralConvolution(nn.Module):
    def setup(self,
              in_channels,
              out_channels,
              n_modes,
              rng_key=jax.random.PRNGKey(42),
              initializer=nn.initializers.xavier_normal()
              ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = (n_modes,) if isinstance(n_modes, int) else n_modes
        self.weights = initializer(rng_key, )
        



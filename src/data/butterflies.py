import jax.numpy as jnp

from .shape import Shape

class Butterfly(Shape):
    def __init__(self, load_name: str):
        self.pts = jnp.load("/Users/vbd402/Documents/Projects/scoreoperator/src/data/raw/"+f"{load_name}.npy")

    def sample(self, n_samples: int):
        if n_samples >= len(self.pts):
            return self.pts
        
        indices = jnp.linspace(0, len(self.pts)-1, n_samples, dtype=jnp.int32)
        return self.pts[indices]
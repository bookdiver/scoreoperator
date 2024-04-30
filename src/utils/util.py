import jax
import jax.numpy as jnp
import functools

from ..data.butterflies import Butterfly
from ..data.synthetic_shapes import Circle
from .trainer import Trainer

class Approximator:

    def __init__(self, trainer: Trainer, approximator_type: str):
        self.approximator_type = approximator_type
        self.trainer = trainer

    def __call__(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        x_ = jnp.expand_dims(x.reshape(-1, 2), axis=0)
        t_ = jnp.expand_dims(jnp.asarray(t), axis=0)
        out = self.trainer.infer_model((x_, t_))
        out = out.squeeze().flatten()
        return out
    
    def forward(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return self.__call__(t, x)

def cpu_run(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with jax.default_device(jax.devices("cpu")[0]):
            return func(*args, **kwargs)
    return wrapper

def get_data(data_type: str, name: str = None):
    if data_type.lower() == "butterfly":
        data_cls = Butterfly
    elif data_type.lower() == "circle":
        data_cls = Circle
    return data_cls(name)
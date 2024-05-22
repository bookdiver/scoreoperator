import jax
import functools
from contextlib import contextmanager

from ..data.butterflies import Butterfly
from ..data.synthetic import Circle

@contextmanager
def use_cpu_backend():
    original_backend = jax.config.read("jax_platform_name")
    jax.config.update("jax_platform_name", "cpu")

    try:
        yield
    finally:
        jax.config.update("jax_platform_name", original_backend)

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
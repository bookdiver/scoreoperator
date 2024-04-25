import jax
import functools

def cpu_run(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with jax.default_device(jax.devices("cpu")[0]):
            return func(*args, **kwargs)
    return wrapper


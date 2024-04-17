import yaml
import jax
import functools

from ml_collections import ConfigDict

def load_config(path: str) -> ConfigDict:
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return ConfigDict(config)

def save_config(config: ConfigDict, path: str) -> None:
    with open(path, 'w') as file:
        yaml.dump(config.to_dict(), file)

def cpu_run(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with jax.default_device(jax.devices("cpu")[0]):
            return func(*args, **kwargs)
    return wrapper
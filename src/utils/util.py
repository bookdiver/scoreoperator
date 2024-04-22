import yaml
import jax
import functools
import ml_collections

class ConfigDict(ml_collections.ConfigDict):

    def to_dict(self):
        """Converts this ConfigDict to a dictionary, preserving the original data types of lists and tuples."""
        def custom_convert(obj):
            if isinstance(obj, ConfigDict):
                # Recursively convert items if they are MyConfigDict instances.
                return {k: custom_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                # Convert each item in the list recursively.
                return [custom_convert(v) for v in obj]
            elif isinstance(obj, tuple):
                # Convert each item in the tuple recursively and ensure the output is a tuple.
                return tuple(custom_convert(v) for v in obj)
            else:
                # Return other types unchanged.
                return obj

        # Apply the custom conversion logic starting from self.
        return custom_convert(self)

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


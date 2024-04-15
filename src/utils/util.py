import yaml
from ml_collections import ConfigDict

def load_config(path: str) -> ConfigDict:
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return ConfigDict(config)

def save_config(config: ConfigDict, path: str) -> None:
    with open(path, 'w') as file:
        yaml.dump(config.to_dict(), file)
import ml_collections

def get_default_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.sde = ml_collections.ConfigDict()
    config.diffusion = ml_collections.ConfigDict()
    config.model = ml_collections.ConfigDict()
    config.training = ml_collections.ConfigDict()
    
    return config
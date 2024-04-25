from .default_config import get_default_config

def get_butterflies_eulerian_config():
    config = get_default_config()

    config.sde_name = "eulerian"

    config.sde.sigma = 0.1
    config.sde.kappa = 0.1

    config.model.lifting_dim = 16
    config.model.co_dims_fmults = [1, 2, 4, 4]
    config.model.n_modes_per_layer = [16, 8, 6, 6]
    config.model.act = "gelu"
    config.model.norm = "batch"

    return config
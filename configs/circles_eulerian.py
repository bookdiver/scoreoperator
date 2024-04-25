from .default_config import get_default_config

def get_circles_eulerian_config():
    config = get_default_config()

    config.sde_name = "eulerian"

    config.sde.sigma = 0.2
    config.sde.kappa = 0.5
    config.model.out_co_dim = 2
    config.model.lifting_dim = 16
    config.model.co_dims_fmults = [1, 2, 4, 4]
    config.model.n_modes_per_layer = [8, 6, 4, 4]
    config.model.act = "gelu"
    config.model.norm = "batch"
    return config
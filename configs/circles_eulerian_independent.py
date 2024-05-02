from .default_config import get_default_config

def get_circles_eulerian_independent_config():
    config = get_default_config()

    config.sde_name = "eulerian_independent"

    config.sde.sigma = 0.2
    config.sde.kappa = 0.1
    config.sde.grid_range = (-2.0, 2.0)
    config.sde.grid_sz = 40

    config.model.out_co_dim = 2
    config.model.lifting_dim = 16
    config.model.co_dims_fmults = [1, 2, 4, 4]
    config.model.n_modes_per_layer = [8, 6, 4, 4]
    config.model.act = "gelu"
    config.model.norm = "batch"
    return config
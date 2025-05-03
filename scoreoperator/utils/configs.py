from ml_collections import ConfigDict

from scoreoperator.utils.data import DataFactory

config = ConfigDict()
config.x0_config = ConfigDict()
config.xT_config = ConfigDict()
config.sde_config = ConfigDict()
config.model_config = ConfigDict()
config.train_config = ConfigDict()

def load_toy_brownian_config():
    config.data.data_type = "toy"
    config.data.x0_config = ConfigDict()
    config.data.x0_config.a = 1.0
    config.data.x0_config.c = 0.0
    config.data.xT_config = ConfigDict()
    config.data.xT_config.a = -1.0
    config.data.xT_config.c = 0.0
    
    config.db.sde_type = "brownian"
    config.db.sde_config = ConfigDict()
    config.db.sde_config.T = 1.0
    config.db.sde_config.dt = 0.02
    config.db.sde_config.sigma = 0.1
    config.db.sde_config.bm_shape = (8, 1)
    
    config.model_type = "CTUNO1D"
    config.model.model_config = ConfigDict()
    config.model.model_config.d_u = 1
    config.model.model_config.d_v = 1
    config.model.model_config.c = 32
    config.model.model_config.d_ls = (16, 32, 64)
    config.model.model_config.modes_ls = (5, 4)
    
    config.training.dir = "../ckpts/toy_brownian"
    config.training.seed = 42
    config.training.lr = 1.0e-3
    config.training.batch_size = 32
    config.training.n_iters = 10000
    config.training.log_freq = 1000
    
    return config

def load_ellipse_cylindrical_brownian_config():
    s0 = DataFactory.create(
        "ellipse",
        {
            "a": 1.25,
            "b": 0.85,
            "shift_x": 0.0,
            "shift_y": 0.0
        }
    )
    sT = DataFactory.create(
        "ellipse",
        {
            "a": 1.5,
            "b": 0.5,
            "shift_x": 0.0,
            "shift_y": 0.0
        }
    )
    
    config.sde_type = "cylindrical_brownian"
    config.sde_config.x0 = DataFactory.create("Zero", {})
    config.sde_config.xT = sT - s0
    config.sde_config.T = 1.0
    config.sde_config.sigma = 0.1
    
    config.model_type = "CTUNO1D"
    config.model_config.d_u = 2
    config.model_config.d_v = 2
    config.model_config.c = 16
    config.model_config.d_ls = (16, 32, 32)
    config.model_config.modes_ls = (8, 6)
    
    config.train_config.dir = "./ckpts/ellipse_cylindrical_brownian"
    config.train_config.seed = 42
    config.train_config.train_x_shape = (16, 2)
    config.train_config.train_t_shape = (100,)
    config.train_config.train_w_shape = (16, 2)
    config.train_config.lr = 1.0e-3
    config.train_config.batch_size = 32
    config.train_config.n_iters = 5000
    config.train_config.log_freq = 1000
    
    return config
    
    
def load_ellipsoid_brownian_config():
    config.data_type = "ellipsoid"
    config.x0_config.a = 0.8
    config.x0_config.b = 0.8
    config.x0_config.c = 0.8
    config.x0_config.shift_x = 0.0
    config.x0_config.shift_y = 0.0
    config.x0_config.shift_z = 0.0
    config.xT_config.a = 0.5
    config.xT_config.b = 0.5
    config.xT_config.c = 0.5
    config.xT_config.shift_x = 0.0
    config.xT_config.shift_y = 0.0
    config.xT_config.shift_z = 0.0
    
    config.sde_type = "brownian"
    config.sde_config.T = 1.0
    config.sde_config.dt = 0.02
    config.sde_config.sigma = 0.1
    config.sde_config.bm_shape = (16, 16, 3)
    
    config.model_type = "CTUNO2D"
    config.model_config.d_u = 3
    config.model_config.d_v = 3
    config.model_config.c = 32
    config.model_config.d_ls = (16, 32, 32)
    config.model_config.modes_ls = (8, 6)
    
    config.train_config.dir = "../ckpts/ellipsoid_brownian"
    config.train_config.seed = 42
    config.train_config.train_x_shape = (16, 2)
    config.train_config.train_t_shape = (100,)
    config.train_config.train_w_shape = (16, 2)
    config.train_config.lr = 1.0e-3
    config.train_config.batch_size = 16
    config.train_config.n_iters = 5000
    config.train_config.log_freq = 1000
    
    return config

def load_butterfly_kunita_config():
    s0 = DataFactory.create(
        "butterfly",
        {"name": "archon_apollinus"}
    )
    sT = DataFactory.create(
        "butterfly",
        {"name": "battus_polydamas"}
    )
    
    config.sde_type = "kunita_flow"
    config.sde_config.x0 = DataFactory.create("Zero", {})
    config.sde_config.xT = sT - s0
    config.sde_config.T = 1.0
    config.sde_config.k_alpha = 0.12
    config.sde_config.k_sigma = 0.3
    
    config.model_type = "CTUNO1D"
    config.model_config.d_u = 2
    config.model_config.d_v = 2
    config.model_config.c = 32
    config.model_config.d_ls = (16, 32, 64, 64)
    config.model_config.modes_ls = (16, 8, 6)
    
    config.train_config.dir = "ckpts/butterfly_kunita_archon_apollinus"
    config.train_config.seed = 42
    config.train_config.train_x_shape = (30, 2)
    config.train_config.train_t_shape = (100,)
    config.train_config.train_w_shape = (100, 100, 2)
    config.train_config.lr = 1.0e-3
    config.train_config.batch_size = 16
    config.train_config.n_iters = 20000
    config.train_config.log_freq = 1000
    
    return config
    
def load_config(experiment_name):
    if experiment_name == "toy_brownian":
        return load_toy_brownian_config()
    elif experiment_name == "ellipse_cylindrical_brownian":
        return load_ellipse_cylindrical_brownian_config()
    elif experiment_name == "ellipsoid_brownian":
        return load_ellipsoid_brownian_config()
    elif experiment_name == "butterfly_kunita":
        return load_butterfly_kunita_config()
    else:
        raise ValueError(f"Unknown experiment name: {experiment_name}")
    
def update_config(update_dict):
    """
    Update the global config with values from a dictionary.
    
    Args:
        update_dict (dict): Dictionary with configuration updates
    
    Returns:
        The updated config object
    """
    def _update_config_dict(config_dict, update_dict):
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in config_dict and isinstance(config_dict[key], ConfigDict):
                _update_config_dict(config_dict[key], value)
            else:                                      
                config_dict[key] = value
    
    _update_config_dict(config, update_dict)
    return config
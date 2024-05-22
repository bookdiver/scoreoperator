from .default_config import get_default_config

def get_circles_brownian_config():
    config = get_default_config()

    config.sde.name = "brownian"
    config.sde.sigma = 0.1

    config.diffusion.matching_obj = "g2score"
    config.diffusion.dt = 2e-2

    config.model.out_co_dim = 2
    config.model.lifting_dim = 16
    config.model.co_dims_fmults = [1, 2, 4]
    config.model.n_modes_per_layer = [8, 6, 4]
    config.model.act = "gelu"
    config.model.norm = "batch"

    config.training.seed = 42
    config.training.n_train_pts = 16
    config.training.n_test_pts = 64
    config.training.learning_rate = 1e-3
    config.training.batch_size = 16
    config.training.optimizer_name = "adam"
    config.training.warmup_steps = 500
    config.training.train_num_epochs = 100
    config.training.train_num_steps_per_epoch = 100
    return config

def get_circles_eulerian_config():
    config = get_default_config()

    config.sde.name = "eulerian"
    config.sde.sigma = 0.1
    config.sde.kappa = 0.2

    config.diffusion.matching_obj = "g2score"
    config.diffusion.dt = 2e-2

    config.model.out_co_dim = 2
    config.model.lifting_dim = 16
    config.model.co_dims_fmults = [1, 2, 4, 4]
    config.model.n_modes_per_layer = [8, 6, 4, 4]
    config.model.act = "gelu"
    config.model.norm = "batch"

    config.training.seed = 42
    config.training.n_train_pts = 16
    config.training.n_test_pts = 64
    config.training.learning_rate = 1e-3
    config.training.batch_size = 16
    config.training.optimizer_name = "adam"
    config.training.warmup_steps = 1000
    config.training.train_num_epochs = 100
    config.training.train_num_steps_per_epoch = 100
    return config

def get_circles_eulerian_independent_config():
    config = get_default_config()

    config.sde.name = "eulerian_independent"
    config.sde.sigma = 0.2
    config.sde.kappa = 0.1
    config.sde.grid_range = (-2.0, 2.0)
    config.sde.grid_sz = 40

    config.diffusion.matching_obj = "g2score"
    config.diffusion.dt = 2e-2

    config.model.out_co_dim = 2
    config.model.lifting_dim = 16
    config.model.co_dims_fmults = [1, 2, 4, 4]
    config.model.n_modes_per_layer = [8, 6, 4, 4]
    config.model.act = "gelu"
    config.model.norm = "batch"

    config.training.seed = 42
    config.training.n_train_pts = 16
    config.training.n_test_pts = 64
    config.training.learning_rate = 1e-3
    config.training.batch_size = 16
    config.training.optimizer_name = "adam"
    config.training.warmup_steps = 500
    config.training.train_num_epochs = 50
    config.training.train_num_steps_per_epoch = 100
    return config

def get_butterflies_eulerian_config():
    config = get_default_config()

    config.sde.name = "eulerian"
    config.sde.sigma = 0.03
    config.sde.kappa = 0.08

    config.diffusion.matching_obj = "g2score"
    config.diffusion.dt = 1e-2

    config.model.out_co_dim = 2
    config.model.lifting_dim = 16
    config.model.co_dims_fmults = [1, 2, 4, 4]
    config.model.n_modes_per_layer = [16, 8, 6, 6]
    config.model.act = "gelu"
    config.model.norm = "batch"

    config.training.seed = 42
    config.training.n_train_pts = 32
    config.training.n_test_pts = 128
    config.training.learning_rate = 1e-3
    config.training.batch_size = 16
    config.training.optimizer_name = "adam"
    config.training.warmup_steps = 2000
    config.training.train_num_epochs = 100
    config.training.train_num_steps_per_epoch = 200
    return config

def get_butterflies_eulerian_independent_config():
    config = get_default_config()

    config.sde.name = "eulerian_independent"
    config.sde.sigma = 0.04
    config.sde.kappa = 0.02
    config.sde.grid_range = (-0.5, 1.5)
    config.sde.grid_sz = 50

    config.diffusion.matching_obj = "g2score"
    config.diffusion.dt = 1e-2

    config.model.out_co_dim = 2
    config.model.lifting_dim = 16
    config.model.co_dims_fmults = [1, 2, 4, 4]
    config.model.n_modes_per_layer = [16, 8, 6, 6]
    config.model.act = "gelu"
    config.model.norm = "batch"

    config.training.seed = 42
    config.training.n_train_pts = 32
    config.training.n_test_pts = 128
    config.training.learning_rate = 1e-3
    config.training.batch_size = 16
    config.training.optimizer_name = "adam"
    config.training.warmup_steps = 500
    config.training.train_num_epochs = 50
    config.training.train_num_steps_per_epoch = 100
    return config

def get_quadratic_brownian_config():
    config = get_default_config()

    config.sde.name = "brownian"
    config.sde.sigma = 0.1

    config.diffusion.matching_obj = "g2score"
    config.diffusion.dt = 2e-2

    config.model.name = "uno"
    config.model.out_co_dim = 1
    config.model.lifting_dim = 16
    config.model.co_dims_fmults = [1, 2, 4]
    config.model.n_modes_per_layer = [6, 4, 2]
    config.model.act = "gelu"
    config.model.norm = "batch"

    config.training.seed = 42
    config.training.n_train_pts = 8
    config.training.n_test_pts = 128
    config.training.learning_rate = 1e-3
    config.training.batch_size = 16
    config.training.optimizer_name = "adam"
    config.training.warmup_steps = 500
    config.training.train_num_epochs = 100
    config.training.train_num_steps_per_epoch = 100
    return config

def get_quadratic_ou_config():
    config = get_default_config()

    config.sde.name = "ou"
    config.sde.sigma = 0.5
    config.sde.theta = 0.1

    config.diffusion.matching_obj = "score"
    config.diffusion.dt = 2e-2

    config.model.out_co_dim = 1
    config.model.lifting_dim = 16
    config.model.co_dims_fmults = [1, 2, 4]
    config.model.n_modes_per_layer = [6, 4, 2]
    config.model.act = "gelu"
    config.model.norm = "batch"

    config.training.seed = 42
    config.training.n_train_pts = 8
    config.training.n_test_pts = 128
    config.training.learning_rate = 1e-3
    config.training.batch_size = 16
    config.training.optimizer_name = "adam"
    config.training.warmup_steps = 500
    config.training.train_num_epochs = 100
    config.training.train_num_steps_per_epoch = 100
    return config


def get_stochastic_heat_config():
    config = get_default_config()

    config.sde.name = "stochastic_heat"
    config.sde.sigma = 0.1
    config.sde.kappa = 1.0
    config.sde.dx = 0.2

    config.diffusion_matching_obj = "g2score"
    config.diffusion.dt = 1e-1

    config.training.seed = 42
    config.training.n_train_pts = 8
    config.training.n_test_pts = 128
    config.training.learning_rate = 1e-3
    config.training.batch_size = 8
    config.training.optimizer_name = "adam"
    config.training.warmup_steps = 500
    config.training.train_num_epochs = 100
    config.training.train_num_steps_per_epoch = 10
    return config



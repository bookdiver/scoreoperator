import ml_collections

def get_default_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.sde_name = "brownian"

    config.sde = sde = ml_collections.ConfigDict()
    sde.sigma = 1.0

    config.diffusion = diffusion = ml_collections.ConfigDict()
    diffusion.dt = 1e-2
    diffusion.approximator = "score"

    config.model = model = ml_collections.ConfigDict()
    model.out_co_dim = 2
    model.lifting_dim = 16
    model.co_dims_fmults = [1, 2, 4, 4]
    model.n_modes_per_layer = [8, 6, 4, 4]
    model.act = "gelu"
    model.norm = "batch"

    config.training = training = ml_collections.ConfigDict()
    training.n_pts = 16
    training.seed = 0
    training.learning_rate = 1e-3
    training.batch_size = 16
    training.optimizer_name = "adam"
    training.warmup_steps = 1000
    training.train_num_epochs = 100
    training.train_num_steps_per_epoch = 50

    return config
import ml_collections

def get_default_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.sde = ml_collections.ConfigDict()

    config.diffusion = diffusion = ml_collections.ConfigDict()
    diffusion.matching_obj = "score"
    diffusion.dt = 1e-2

    config.model = model = ml_collections.ConfigDict()
    model.in_co_dim = 1
    model.out_co_dim = 1
    model.lifting_dim = 8
    model.co_dims_fmults = [1, 2, 4]
    model.n_modes_per_layer = [8, 6, 4]
    model.act = "gelu"
    model.norm = "batch"

    config.training = training = ml_collections.ConfigDict()
    training.n_train_pts = 16
    training.n_test_pts = 32
    training.seed = 0
    training.learning_rate = 1e-3
    training.batch_size = 16
    training.optimizer_name = "adam"
    training.warmup_steps = 1000
    training.train_num_epochs = 100
    training.train_num_steps_per_epoch = 50

    return config
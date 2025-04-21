import argparse
import os
import jax.numpy as jnp

from bridgeop.utils.trainer import TrainerModule
from bridgeop.utils.data import DataFactory
from bridgeop.utils.configs import load_config
from bridgeop.neuralop.uno import ModelFactory
from bridgeop.diffusion.bridge import DiffusionBridge

def main(args):
    # Get the specific configuration for the chosen experiment
    config = load_config(args.experiment)
    cwd = os.getcwd()
    absolute_path = os.path.join(cwd, config.train_config.dir)
    absolute_path = os.path.normpath(absolute_path)
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)
    
    config.train_config.dir = absolute_path

    # Create data
    x0 = DataFactory.create(
        data_type=config.data_type,
        data_config=config.x0_config,
    )
    xT = DataFactory.create(
        data_type=config.data_type,
        data_config=config.xT_config,
    )
    
    # Create diffusion bridge
    db = DiffusionBridge(config.sde_type, config.sde_config)
    
    # Create model
    model = ModelFactory.create(config.model_type, config.model_config)

    # Initialize trainer
    trainer = TrainerModule(db, model, config.train_config)
    
    # Get initial points
    x0_eval = x0.eval(config.sde_config.x_shape)
    print(f"Train on shape: {x0_eval.shape}")

    trainer.train_model(x0=x0_eval, mode='train')
    # trainer.train_model(x0=jnp.zeros_like(x0_eval), mode='train')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or load a model based on configuration.")
    parser.add_argument('--experiment', type=str, required=True, help='Name of the experiment in the config file')

    args = parser.parse_args()

    main(args)

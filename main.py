import argparse
import os
import jax.numpy as jnp

from scoreoperator.neuralop.uno import ModelFactory
from scoreoperator.diffusion.bridge import DiffusionBridge
from scoreoperator.utils.trainer import TrainerModule
from scoreoperator.utils.data import DataFactory
from scoreoperator.utils.configs import load_config

def main(args):
    # Get the specific configuration for the chosen experiment
    config = load_config(args.experiment)
    cwd = os.getcwd()
    absolute_path = os.path.join(cwd, config.train_config.dir)
    absolute_path = os.path.normpath(absolute_path)
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)

    # Create data
    x0 = DataFactory.create(
        data_type=config.data_type,
        data_config=config.x0_config,
    )
    xT = DataFactory.create(
        data_type=config.data_type,
        data_config=config.xT_config,
    )
    # Get initial points
    x0_eval = x0.eval(config.train_config.train_shape)
    xT_eval = xT.eval(config.train_config.train_shape)
    print(f"Train on shape: {config.train_config.train_shape}")
    
    # Create diffusion bridge
    config.sde_config.x0 = x0_eval
    config.sde_config.xT = xT_eval
    db = DiffusionBridge(config.sde_type, config.sde_config)
    
    # Create model
    model = ModelFactory.create(config.model_type, config.model_config)

    # Initialize trainer
    trainer = TrainerModule(db, model, config.train_config)
    
    
    # Determine training mode based on arguments
    if args.resume_step is not None:
        print(f"Resuming training from step {args.resume_step}")
        trainer.train_model(
            x0=jnp.zeros_like(x0_eval), 
            mode='resume', 
            resume_step=args.resume_step
        )
    else:
        print("Starting training from scratch")
        trainer.train_model(
            x0=jnp.zeros_like(x0_eval), 
            mode='train'
        )

    # trainer.train_model(x0=x0_eval, mode='train')
    # trainer.train_model(x0=jnp.zeros_like(x0_eval), mode='train')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or load a model based on configuration.")
    parser.add_argument('--experiment', type=str, required=True, help='Name of the experiment in the config file')
    parser.add_argument('--resume-step', type=int, help='Resume training from a specific checkpoint step')

    args = parser.parse_args()

    main(args)

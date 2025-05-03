import argparse
import os

from scoreoperator.neuralop.uno import ModelFactory
from scoreoperator.diffusion.bridge import DiffusionBridge
from scoreoperator.utils.trainer import TrainerModule
from scoreoperator.utils.configs import load_config

def main(args):
    # Get the specific configuration for the chosen experiment
    config = load_config(args.experiment)
    cwd = os.getcwd()
    absolute_path = os.path.join(cwd, config.train_config.dir)
    absolute_path = os.path.normpath(absolute_path)
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)

    db = DiffusionBridge(
        config.sde_type, 
        config.sde_config
    )
    
    if args.train_x_shape is not None:
        x_shape = args.train_x_shape
    else:
        x_shape = config.train_config.train_x_shape 
    
    db.project(
        x_shape,
        config.train_config.train_t_shape,
        config.train_config.train_w_shape
    )
    
    # Create model
    model = ModelFactory.create(
        config.model_type, 
        config.model_config
    )

    # Initialize trainer
    trainer = TrainerModule(db, model, config.train_config)
    
    # Determine training mode based on arguments
    if args.resume_step is not None:
        print(f"Resuming training from step {args.resume_step}")
        trainer.train_model(
            mode='resume', 
            resume_step=args.resume_step
        )
    else:
        print("Starting training from scratch")
        trainer.train_model( 
            mode='train'
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or load a model based on configuration.")
    parser.add_argument('--experiment', type=str, required=True, help='Name of the experiment in the config file')
    parser.add_argument('--resume-step', type=int, help='Resume training from a specific checkpoint step')
    parser.add_argument('--train-x-shape', type=tuple, help='Shape of the training data')
    args = parser.parse_args()

    main(args)

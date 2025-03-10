import yaml
import argparse
import os
import jax.numpy as jnp
from bridgeop.utils.trainer import TrainerModule
from bridgeop.utils.data import DataFactory

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Get the specific configuration for the chosen experiment
    experiment_config = config.get(args.experiment)
    cwd = os.getcwd()
    absolute_path = os.path.join(cwd, experiment_config["training"]["dir"])
    absolute_path = os.path.normpath(absolute_path)
    experiment_config["training"]["dir"] = absolute_path + "_neuralop"
    if experiment_config is None:
        raise ValueError(f"Experiment '{args.experiment}' not found in config file.")

    # Create data
    data_config = experiment_config.get('data', {})
    data = DataFactory.create(data_config)
    
    if args.n_pts is not None:
        experiment_config['training']['n_train_pts'] = (args.n_pts, ) * len(experiment_config['training']['n_train_pts'])
        if "eulerian" not in args.experiment:
            experiment_config['diffusion_bridge']['sde_kwargs']['W_shape'] = (args.n_pts, ) * (len(experiment_config['diffusion_bridge']['sde_kwargs']['W_shape']) - 1) + (experiment_config['diffusion_bridge']['sde_kwargs']['W_shape'][-1], )
            
    # Initialize trainer
    trainer = TrainerModule(experiment_config)
    
    # Get initial points
    n_train_pts = experiment_config['training']['n_train_pts']
    initial_points = data.eval(n_train_pts)['x0']
    
    # Train or load model
    if args.mode == 'train':
        trainer.train_model(x0=initial_points, mode='train')
    elif args.mode == 'resume':
        trainer.train_model(x0=initial_points, mode='resume', step=args.step)
    elif args.mode == 'pretrained':
        trainer.train_model(x0=initial_points, mode='pretrained', step=args.step)
    else:
        raise ValueError("Invalid mode. Choose 'train', 'resume', or 'pretrained'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or load a model based on configuration.")
    parser.add_argument('--config', type=str, default='./config.yaml', help='Path to the configuration file')
    parser.add_argument('--experiment', type=str, required=True, help='Name of the experiment in the config file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'resume', 'pretrained'], help='Mode of operation')
    parser.add_argument('--step', type=int, help='Step to resume from or load pretrained model (required for resume and pretrained modes)')
    
    parser.add_argument('--n_pts', type=int, default=None, help='Number of points to train on')
    
    args = parser.parse_args()
    
    if args.mode in ['resume', 'pretrained'] and args.step is None:
        parser.error("--step is required when mode is 'resume' or 'pretrained'")
    
    main(args)

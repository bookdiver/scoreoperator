import os
import logging
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from ml_collections import ConfigDict
from configs.configs import *
from src.data.synthetic import Circle, Quadratic
from src.data.butterfly import Butterfly
from src.utils.trainer import TrainerModule, Model
from src.utils.plotting import plot_trajectories

PATH = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()

parser.add_argument("--experiment", type=str, default="circle", choices=["circle", "butterfly", "quadratic"])
parser.add_argument("--target", type=str, default="parnassius_honrathi")
parser.add_argument("--start", type=str, default="papilio_polytes")
parser.add_argument("--sde", type=str, default="brownian", choices=["brownian", "eulerian", "eulerian_independent", "ou"])
parser.add_argument("--matching_obj", type=str, default="score", choices=["score", "gscore", "g2score"])

parser.add_argument("--seed", type=int, nargs="?", default=False, help="Seed for reproducibility.")
parser.add_argument("--n_train_pts", type=int, nargs="?", default=False, help="Number of sampled points of the function for training.")
parser.add_argument("--n_test_pts", type=int, nargs="?", default=False, help="Number of sampled points of the function for testing.")
parser.add_argument("--lr", type=float, nargs="?", default=False, help="Learning rate for training.")
parser.add_argument("--batch_sz", type=int, nargs="?", default=False, help="Batch size for training.")
parser.add_argument("--n_epochs", type=int, nargs="?", default=False, help="Number of epochs for training.")
parser.add_argument("--n_steps_per_epoch", type=int, nargs="?", default=False, help="Number of steps per epoch for training.")

parser.add_argument("--debug", action="store_true", help="Set logging level to debug.")
parser.add_argument("--eval", action="store_true", help="Evaluate the model and plot the trajectories.")
parser.add_argument("--show_samples", action="store_true", help="Show samples from the training set, but do not train the model.")

args = parser.parse_args()

def read_config(experiment: str, sde: str) -> ConfigDict:
    if experiment == "circle":
        if sde == "brownian":
            return get_circles_brownian_config()
        elif sde == "eulerian":
            return get_circles_eulerian_config()
        elif sde == "eulerian_independent":
            return get_circles_eulerian_independent_config()
    elif experiment == "butterfly":
        if sde == "eulerian":
            return get_butterflies_eulerian_config()
        elif sde == "eulerian_independent":
            return get_butterflies_eulerian_independent_config()
    elif experiment == "quadratic":
        if sde == "brownian":
            return get_quadratic_brownian_config()
        elif sde == "ou":
            return get_quadratic_ou_config()
    elif experiment == "heat":
        return get_stochastic_heat_config()

def main():
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.experiment == "circle":
        X0 = Circle(r=1.0)
        XT = Circle(r=1.5)
    
    elif args.experiment == "butterfly":
        X0 = Butterfly(args.start, interp=300, interp_method="linear")
        XT = Butterfly(args.target, interp=300, interp_method="linear")

    elif args.experiment == "quadratic":
        X0 = Quadratic(a=1.0, shift=0.0)
        XT = Quadratic(a=-1.0, shift=0.0)

    config = read_config(args.experiment, args.sde)
    config.sde.X0 = X0
    
    config.diffusion.matching_obj = args.matching_obj

    config.training.seed = args.seed if args.seed else config.training.seed
    config.training.n_train_pts = args.n_train_pts if args.n_train_pts else config.training.n_train_pts
    config.training.n_test_pts = args.n_test_pts if args.n_test_pts else config.training.n_test_pts
    config.training.train_num_epochs = args.n_epochs if args.n_epochs else config.training.train_num_epochs
    config.training.train_num_steps_per_epoch = args.n_steps_per_epoch if args.n_steps_per_epoch else config.training.train_num_steps_per_epoch
    config.training.dir = os.path.join(PATH, "results", f"{args.experiment}_{args.sde}_{args.matching_obj}_{args.start}")

    if not os.path.exists(config.training.dir):
        os.mkdir(config.training.dir)

    trainer = TrainerModule(config)
    logging.info(f"Training {args.experiment} with {args.sde} SDE and {args.matching_obj} matching objective")

    if args.show_samples:
        xss, *_ = next(trainer.dataloader)
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        for i in range(9):
            xs = xss[i].reshape(xss.shape[1], xss.shape[2]//X0.co_dim, X0.co_dim)
            x0 = X0.sample(config.training.n_train_pts)
            logging.debug(f"x0: {x0.shape}")
            xs = xs + x0[None, ...]
            plot_trajectories(dim=X0.co_dim,
                              ax=axes[i], 
                              traj=xs, 
                              target=x0, 
                              cmap_name="rainbow")
        fig.savefig(os.path.join(config.training.dir, "samples.png"))
        logging.info(f"Samples saved in {config.training.dir}")
        return 0
    
    trainer.train_model(pretrained=args.eval)
    model = Model(trainer)

    y0 = (XT.sample(config.training.n_test_pts) - X0.sample(config.training.n_test_pts)).flatten()
    ts = jnp.linspace(0.0, 1.0, 200)
    ys = trainer.diffuser.solve_reverse_bridge_sde(rng_key=jax.random.PRNGKey(args.seed), x0=y0, ts=ts, model=model)
    ys = ys.reshape(ys.shape[0], ys.shape[1]//X0.co_dim, X0.co_dim)
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    plot_trajectories(dim=X0.co_dim,
                      ax=ax1,
                      traj=ys + X0.sample(config.training.n_test_pts)[None, ...],
                      target=X0.sample(config.training.n_test_pts),
                      cmap_name="rainbow")
    fig1.savefig(os.path.join(config.training.dir, f"trajectories_{config.training.n_test_pts}.png"))
    plt.close(fig1)

    if not args.eval:
        train_losses = np.loadtxt(os.path.join(config.training.dir, "train_losses.txt"))
        eval_losses = np.loadtxt(os.path.join(config.training.dir, "eval_losses.txt"))
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
        ax2.plot(train_losses, label="Train loss")
        ax2.plot(eval_losses, label="Eval loss")
        fig2.savefig(os.path.join(config.training.dir, "losses.png"))

if __name__ == "__main__":
    main()

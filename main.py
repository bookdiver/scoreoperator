import os
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from configs.circles_brownian import get_circles_brownian_config
from configs.circles_eulerian import get_circles_eulerian_config
from configs.butterflies_eulerian import get_butterflies_eulerian_config

from src.data.synthetic_shapes import Circle
from src.data.butterflies import Butterfly
from src.utils.trainer import TrainerModule, Model
from src.utils.plotting import plot_trajectories

PATH = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()

parser.add_argument("--experiment", type=str, default="circle", choices=["circle", "butterfly"])
parser.add_argument("--sde", type=str, default="brownian", choices=["brownian", "eulerian"])
parser.add_argument("--matching_obj", type=str, default="score", choices=["score", "gscore", "g2score"])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n_train_points", type=int, default=16)
parser.add_argument("--n_test_points", type=int, default=64)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--n_steps_per_epoch", type=int, default=100)

args = parser.parse_args()

def main():
    if args.experiment == "circle":
        shape1 = Circle(r=1.0)
        shape2 = Circle(r=1.5)

        if args.sde == "brownian":
            config = get_circles_brownian_config()
            config.training.n_pts = args.n_train_points
            config.diffusion.matching_obj = args.matching_obj
        elif args.sde == "eulerian":
            config = get_circles_eulerian_config()
            config.sde.s0 = shape1
            config.training.n_pts = args.n_train_points
            config.diffusion.matching_obj = args.matching_obj
    
    elif args.experiment == "butterfly":
        shape1 = Butterfly("example_butterfly1", interpolation=512, interpolation_type="linear")
        shape2 = Butterfly("example_butterfly2", interpolation=512, interpolation_type="linear")

        if args.sde == "eulerian":
            config = get_butterflies_eulerian_config()
            config.sde.s0 = shape1
            config.training.n_pts = args.n_train_points
            config.diffusion.matching_obj = args.matching_obj
    
    config.training.seed = args.seed
    config.training.train_num_epochs = args.n_epochs
    config.training.train_num_steps_per_epoch = args.n_steps_per_epoch
    config.training.dir = os.path.join(PATH, "results", f"{args.experiment}_{args.sde}_{args.matching_obj}")

    if not os.path.exists(config.training.dir):
        os.mkdir(config.training.dir)

    trainer = TrainerModule(config)
    print(f"Training {args.experiment} with {args.sde} SDE and {args.matching_obj} matching objective")
    trainer.train_model()
    model = Model(trainer)

    x0 = (shape2.sample(args.n_test_points) - shape1.sample(args.n_test_points)).flatten()
    ts = jnp.linspace(0.0, 1.0, 200)
    xs = trainer.diffuser.solve_reverse_bridge_sde(rng_key=jax.random.PRNGKey(args.seed), x0=x0, ts=ts, model=model)
    xs = xs.reshape(-1, args.n_test_points, 2)
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    plot_trajectories(ax1,
                      xs + shape1.sample(args.n_test_points)[None, ...],
                      target=shape1.sample(args.n_test_points),
                      plot_target=True,
                      cmap_name="rainbow")
    fig1.savefig(os.path.join(config.training.dir, "trajectories.png"))
    plt.close(fig1)

    train_losses = np.loadtxt(os.path.join(config.training.dir, "train_losses.txt"))
    eval_losses = np.loadtxt(os.path.join(config.training.dir, "eval_losses.txt"))
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
    ax2.plot(train_losses, label="Train loss")
    ax2.plot(eval_losses, label="Eval loss")
    fig2.savefig(os.path.join(config.training.dir, "losses.png"))

if __name__ == "__main__":
    main()

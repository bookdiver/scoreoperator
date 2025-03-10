import argparse
import os

import time

import jax
import jax.numpy as jnp
from flax.training import checkpoints

import diffusion_bridge as db
import sdes, utils
from score_unet import ScoreUNet

from bridgeop.utils.data import DataFactory

parser = argparse.ArgumentParser()
parser.add_argument("--n_bases", type=int, default=32)
parser.add_argument("--n_pts", type=int, default=100)


def run(n_bases, n_pts):
    save_path = f"../../ckpts/ellipse_brownian_baseline_{n_bases}_fourier_{n_pts}_pts"
    cwd = os.getcwd()
    absolute_path = os.path.join(cwd, save_path)
    absolute_path = os.path.normpath(absolute_path)
    
    sde_config = {
        "T": 1.0,
        "Nt": 100,
        "dim": 2,
        "n_bases": n_bases,
        "sigma": 0.1,
    }

    bm_sde = sdes.brownian_sde(**sde_config)

    network = {
        "output_dim": 2 * bm_sde.dim * bm_sde.n_bases,
        "time_embedding_dim": bm_sde.dim * bm_sde.n_bases * 4,
        "init_embedding_dim": bm_sde.dim * bm_sde.n_bases * 4,
        "act_fn": "silu",
        "encoder_layer_dims": [
            bm_sde.dim * bm_sde.n_bases * 8,
            bm_sde.dim * bm_sde.n_bases * 4,
            bm_sde.dim * bm_sde.n_bases * 2,
            bm_sde.dim * bm_sde.n_bases * 1,
        ],
        "decoder_layer_dims": [
            bm_sde.dim * bm_sde.n_bases * 1,
            bm_sde.dim * bm_sde.n_bases * 2,
            bm_sde.dim * bm_sde.n_bases * 4,
            bm_sde.dim * bm_sde.n_bases * 8,
        ],
        "batchnorm": True,
    }

    training = {
        "batch_size": 32,
        "load_size": 3200,
        "num_epochs": 100,
        "learning_rate": 2e-3,
        "warmup_steps": 500,
    }

    key = jax.random.PRNGKey(42)

    neural_net = ScoreUNet
    data_config = {
        "type": "ellipse",
        "start_point": {
            "a": 1.25,
            "b": 0.85,
        },
        "end_point": {
            "a": 1.5,
            "b": 0.5,
        },
    }
    data = DataFactory.create(data_config)
    target = data.eval((n_pts, ))["x0"]
    target = utils.fourier_coefficients(target, n_bases)

    def target_sampler(key, num_batches):
        initial_vals = jnp.tile(target, reps=(num_batches, 1, 1, 1))
        return initial_vals

    train_key = jax.random.split(key, 2)[0]
    start_time = time.time()
    score_state_p = db.learn_p_score(
        bm_sde, target_sampler, train_key, aux_dim=2, **training, net=neural_net, network_params=network
    )
    end_time = time.time()
    t = end_time - start_time
    print(f"Time taken: {t} seconds")

    try:
        ckpt_path = checkpoints.save_checkpoint(
            ckpt_dir=absolute_path,
            target={
                "params": score_state_p.params,
                "batch_stats": score_state_p.batch_stats,
            },
            step=10000,
            prefix="checkpoint_",
            overwrite=True
        )
        print(f"Model checkpoint saved successfully to {ckpt_path}")
    except Exception as e:
        print(f"Failed to save model checkpoint: {str(e)}")


if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(f"./ckpts/ellipse_brownian_baseline_{args.n_bases}_fourier_{args.n_pts}_pts"):
        os.makedirs(f"./ckpts/ellipse_brownian_baseline_{args.n_bases}_fourier_{args.n_pts}_pts")

    run(args.n_bases, args.n_pts)
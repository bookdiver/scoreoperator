import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from flax.training import checkpoints
import diffusion_bridge as db
import sdes, utils
from score_unet import ScoreUNet

from bridgeop.utils.data import DataFactory
def load_ckpt(sde_config, network_config, training_config, save_path):
    ckpt = checkpoints.restore_checkpoint(ckpt_dir=save_path, 
                                          target=None,
                                          step=10000,
                                          prefix="checkpoint_")
    key = jax.random.PRNGKey(42)
    score_net = ScoreUNet(**network_config)
    num_batches_per_epoch = int(training_config["load_size"] / training_config["batch_size"])

    state = utils.create_train_state(
        model=score_net,
        key=key,
        input_shapes=[
            (training_config["batch_size"], 2 * sde_config["n_bases"] * sde_config["dim"]),
            (training_config["batch_size"], 1),
        ],
        learning_rate=training_config["learning_rate"],
        warmup_steps=training_config["warmup_steps"],
        decay_steps=training_config["num_epochs"] * num_batches_per_epoch,
    )

    # Update params and batch_stats according to ckpt
    state = state.replace(
        params=ckpt["params"],
        batch_stats=ckpt["batch_stats"]
    )
    
    return state


if __name__ == "__main__":
    # n_bases = 4

    def restore_for_bases(n_bases):
        save_path = f"../../ckpts/ellipse_brownian_baseline_{n_bases}_fourier_64_pts"
        cwd = os.getcwd()
        absolute_path = os.path.join(cwd, save_path)
        absolute_path = os.path.normpath(absolute_path)
        
        print(save_path)
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

        state = load_ckpt(sde_config, network, training, absolute_path)
        return state, bm_sde


    basis_list = [32]
    for n_bases in basis_list:
        state, bm_sde = restore_for_bases(n_bases)

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
        
        n_eval_pts = [64, 128, 256]
        for n_pts in n_eval_pts:
            target = data.eval((n_pts, ))["x0"]
            target = utils.fourier_coefficients(target, n_bases)

            def forward_score(t0, x0, t, x):
                x0 = jnp.asarray(x0)
                x = jnp.asarray(x)
                return (x0 - x) / (t + 1e-8)

            def error_forward(ts, true_score, trained_score, target, y):
                """mean squared error between true and trained score"""
                true = jax.vmap(true_score, in_axes=(None, None, 0, None))(0, target, ts, y)
                trained = jax.vmap(trained_score, in_axes=(None, 0))(y, ts)
                true_landmark = utils.inverse_fourier(true, n_pts)
                trained_landmark = utils.inverse_fourier(trained, n_pts)
                return jnp.mean(jnp.linalg.norm(true_landmark - trained_landmark, axis=-1))

            ts = jnp.linspace(0, bm_sde.T, 100)
            score_p = utils.score_fn(state)

            error = error_forward(ts[1:], forward_score, score_p, target, target)
            print(f"Error for {n_pts} points: {error}")

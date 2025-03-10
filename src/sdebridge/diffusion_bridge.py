from collections import namedtuple
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow as tf
from einops import rearrange, repeat
from tqdm import tqdm

import sdes
from utils import bse, create_train_state, get_iterable_dataset, invert, mult


def trajectory_generator(
    sde: sdes.SDE,
    key: jax.Array,
    batch_size: int,
    x0_sampler: Callable,
) -> Callable:
    """
    Get the trajectory generator that generates the batched trajectories
    for the forward SDE.
    x0.shape: (1, n_bases, dim) for landmarks,
              (2, n_bases, dim) for Fourier coefficients
    """

    # initial_vals = jnp.tile(x0, reps=(batch_size, 1, 1, 1))  # (B, 1 or 2, n_bases, dim)

    def generator():
        subkey = key
        while True:
            val_key, subkey = jax.random.split(subkey)
            initial_vals = x0_sampler(val_key, batch_size)
            trajs, grads, covs = euler_and_grad_and_cov(
                sde, initial_vals, subkey
            )  # trajs, grads with shape (B, 1 or 2, n_bases, dim), covs with shape (B, n_bases, 1 or 2, n_bases, n_bases)
            yield trajs, grads, covs,

    return generator


def learn_p_score(
    sde: sdes.SDE,
    x0_sampler: Callable,
    key: jax.Array,
    aux_dim: int,
    *,
    batch_size: int,
    load_size: int,
    learning_rate: int,
    warmup_steps: int,
    num_epochs: int,
    net: nn.Module,
    network_params: dict = None,
):
    ts = rearrange(
        repeat(sde.ts[1:], "n -> b n", b=batch_size),
        "b n -> (b n) 1",
        b=batch_size,
    )
    gen = trajectory_generator(sde, key, load_size, x0_sampler)
    return learn_score(
        sde,
        aux_dim,
        gen,
        key,
        ts,
        batch_size=batch_size,
        load_size=load_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_epochs=num_epochs,
        net=net,
        network_params=network_params,
    )


def learn_p_star_score(
    forward_sde: sdes.SDE,
    x0_sampler: Callable,
    key: jax.Array,
    score_p: Callable,
    aux_dim: int,
    *,
    batch_size: int,
    load_size: int,
    learning_rate: int,
    warmup_steps: int,
    num_epochs: int,
    net: nn.Module,
    network_params: dict = None,
):
    ts = rearrange(
        repeat(forward_sde.T - forward_sde.ts[:-1], "n -> b n", b=batch_size),
        # !!! the backward trajectories are in the reverse order, so we need inverted time series.
        "b n -> (b n) 1",
        b=batch_size,
    )
    reverse_sde = sdes.reverse(forward_sde, score_p)
    gen = trajectory_generator(reverse_sde, key, load_size, x0_sampler)

    return learn_score(
        forward_sde,
        aux_dim,
        gen,
        key,
        ts,
        batch_size=batch_size,
        load_size=load_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_epochs=num_epochs,
        net=net,
        network_params=network_params,
    )


def learn_score(
    sde: sdes.SDE,
    aux_dim: int,
    generator: Callable,
    key: jax.Array,
    ts: jax.Array,
    *,
    batch_size: int,
    load_size: int,
    learning_rate: int,
    warmup_steps: int,
    num_epochs: int,
    net: nn.Module,
    network_params: dict = None,
):
    net_params = network_params
    num_batches_per_epoch = int(load_size / batch_size)
    score_net = net(**net_params)

    _, network_key = jax.random.split(key)

    iter_dataset = get_iterable_dataset(
        generator=generator,
        # dtype=(tf.float64, tf.float64, tf.float64),
        dtype=(tf.float32, tf.float32, tf.float32),  # !!! change to float32 for better performance
        shape=[
            (load_size, sde.Nt - 1, aux_dim, sde.n_bases, sde.dim),  # trajs
            (load_size, sde.Nt - 1, aux_dim, sde.n_bases, sde.dim),  # grads
            (load_size, sde.Nt - 1, aux_dim, sde.n_bases, sde.n_bases),  # covs
        ],
    )

    @jax.jit
    def train_step(state, batch: tuple):
        trajs, grads, covs = batch  # (B, Nt, aux_dim, n_bases, dim) and (B, Nt, aux_dim, bm_size, bm_size)
        b = trajs.shape[0]
        n = trajs.shape[1]

        trajs = trajs.reshape((b * n, *trajs.shape[2:]))  # (B*Nt, aux_dim, n_bases, dim)
        grads = grads.reshape((b * n, *grads.shape[2:]))  # (B*Nt, aux_dim, n_bases, dim)
        covs = covs.reshape((b * n, *covs.shape[2:]))  # (B*Nt, aux_dim, n_bases, n_bases)

        def loss_fn(params) -> tuple:
            scores, updates = state.apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                x=trajs,
                t=ts,
                train=True,
                mutable=["batch_stats"],
            )  # score.shape: (B*Nt, aux_dim*n_bases*dim)
            losses = jax.vmap(bse)(scores - grads, covs)  # (B*Nt, )
            loss = 0.5 * jnp.mean(losses, axis=0)
            return loss, updates

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, updates), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=updates["batch_stats"])
        step_key, _ = jax.random.split(state.key)
        state = state.replace(key=step_key)

        return state, loss

    state = create_train_state(
        model=score_net,
        key=network_key,
        input_shapes=[
            (batch_size, aux_dim * sde.n_bases * sde.dim),
            (batch_size, 1),
        ],
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=num_epochs * num_batches_per_epoch,
    )
    pbar = tqdm(
        range(num_epochs),
        desc="Training",
        leave=True,
        unit="epoch",
        total=num_epochs,
    )
    for i in pbar:
        total_loss = 0
        load = next(iter_dataset)
        for b in range(num_batches_per_epoch):
            tmp1, tmp2, tmp3 = load
            batch = (
                tmp1[b * batch_size : (b + 1) * batch_size],
                tmp2[b * batch_size : (b + 1) * batch_size],
                tmp3[b * batch_size : (b + 1) * batch_size],
            )
            state, loss = train_step(state, batch)
            total_loss += loss
        epoch_loss = total_loss / num_batches_per_epoch
        pbar.set_postfix(Epoch=i + 1, loss=f"{epoch_loss:.4f}")

    return state


def batch_matmul(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """Batch matrix multiplication"""
    return jax.vmap(jnp.matmul, in_axes=(0, 0), out_axes=0)(A, B)


def euler_and_grad_and_cov(
    sde: sdes.SDE,
    initial_vals: jnp.ndarray,
    rng_key: jax.Array,
) -> tuple:
    """Euler-Maruyama solver for SDEs

    initial_vals: (B, aux_dim, n_bases, 2)
    """
    b, aux_d, n = initial_vals.shape[:-1]
    state = namedtuple("state", ["x", "grads", "covs", "key"])
    init_state = state(
        x=initial_vals,
        grads=jnp.empty_like(initial_vals),
        covs=jnp.empty((b, aux_d, n, n)),
        key=rng_key,
    )

    def euler_maruyama_step(s: state, time: jnp.ndarray) -> tuple:
        """Euler-Maruyama step, NOTE: all the calculations are over batches"""
        time = jnp.expand_dims(time, axis=-1)
        time = jnp.tile(time, (b, 1))
        step_key, _ = jax.random.split(s.key)
        drift_ = jax.vmap(sde.drift, in_axes=(0, 0))(s.x, time)  # (B, aux_dim, n_bases, dim)

        eps_ = jax.random.normal(step_key, shape=(b, aux_d, *sde.bm_shape))  # (B, aux_dim, bm_size, dim)
        diffusion_ = jax.vmap(sde.diffusion, in_axes=(0, None))(s.x, time)  # (B, aux_dim, n_bases, bm_size)
        diffusion_step = jnp.sqrt(sde.dt) * jax.vmap(mult)(diffusion_, eps_)  # (B, aux_dim, n_bases, dim)

        cov_ = jax.vmap(sdes.cov, in_axes=(None, 0, None))(sde, s.x, time)  # (B, aux_dim, n_bases, n_bases)
        inv_cov = jax.vmap(invert)(cov_)  # (B, aux_dim, n_bases, n_bases)

        grads = -1 / sde.dt * jax.vmap(mult)(inv_cov, diffusion_step)  # (B, aux_dim, n_bases, dim)

        xnew = s.x + drift_ * sde.dt + diffusion_step  # (B, aux_dim, n_bases, dim)
        new_state = state(
            x=xnew,
            grads=grads,
            covs=cov_,
            key=step_key,
        )
        return new_state, (
            s.x,
            s.grads,
            s.covs,
            s.key,
        )

    _, (trajectories, gradients, covariances, step_keys) = jax.lax.scan(
        euler_maruyama_step,
        init=init_state,
        xs=(sde.ts[:-1]),
        length=sde.Nt - 1,
    )
    trajectories = trajectories.swapaxes(0, 1)  # (B, Nt, aux_dim, n_bases, dim)
    gradients = gradients.swapaxes(0, 1)  # (B, Nt, aux_dim, n_bases, dim)
    covariances = covariances.swapaxes(0, 1)  # (B, Nt, aux_dim, n_bases, n_bases)

    return trajectories, gradients, covariances
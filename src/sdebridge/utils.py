from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import linen as nn
from flax.training import train_state


def fourier_coefficients(array, num_bases):
    """Array of shape [..., pts, dim]
    Returns array of shape [..., 2, :num_bases, dim]"""

    complex_coefficients = jnp.fft.rfft(array, norm="forward", axis=-2)[..., :num_bases, :]
    coeffs = jnp.stack([complex_coefficients.real, complex_coefficients.imag], axis=0)
    return coeffs


def inverse_fourier(coefficients, num_pts):
    """Array of shape [..., 2, num_bases, dim]
    Returns array of shape [..., num_pts, dim]"""
    assert coefficients.shape[-2] % 2 == 0
    coeffs_real = coefficients[..., 0, :, :]
    coeffs_im = coefficients[..., 1, :, :]
    complex_coefficients = coeffs_real + 1j * coeffs_im
    return jnp.fft.irfft(complex_coefficients, norm="forward", n=num_pts, axis=-2)


def invert(A: jnp.ndarray) -> jnp.ndarray:
    """A.shape: (aux_dim, n_bases, n_bases)"""
    if A.shape[0] == 1:  # real matrix
        return jnp.expand_dims(jnp.linalg.pinv(A[0], hermitian=True, rcond=None), axis=0)
    else:  # complex matrix but present as 2 real matrices
        C = A[0] + 1j * A[1]
        C_inv = jnp.linalg.pinv(C, hermitian=True, rcond=None)
        return jnp.stack([C_inv.real, C_inv.imag], axis=0)


def mult(A: jnp.ndarray, B: jnp.ndarray, B_conj: bool = False) -> jnp.ndarray:
    """A.shape: (aux_dim, M, n_bases), B.shape: (aux_dim, n_bases, P)
    Returns: (aux_dim, M, P)
    """
    assert A.shape[0] == B.shape[0]
    if A.shape[0] == B.shape[0] == 1:  # real matrix multiplication
        return jnp.expand_dims(jnp.matmul(A[0], B[0]), axis=0)
    else:
        A = A[0] + 1j * A[1]
        if B_conj:
            B = B[0] - 1j * B[1]
        else:
            B = B[0] + 1j * B[1]
        C = jnp.matmul(A, B)
        return jnp.stack([C.real, C.imag], axis=0)


### Dimension helpers
def flatten_batch(x: jax.Array):
    assert len(x.shape) >= 2
    return x.reshape(-1, x.shape[-1])


def unsqueeze(x: jax.Array, axis: int):
    return jnp.expand_dims(x, axis=axis)


### Network helpers


class TrainState(train_state.TrainState):
    key: jax.Array
    batch_stats: any


def create_optimizer(learning_rate: float, warmup_steps: int, decay_steps: int):
    lr_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=0.01 * learning_rate,
    )
    optimizer = optax.chain(
        optax.adam(lr_scheduler),
    )
    return optimizer


def create_train_state(
    model: nn.Module,
    key: jax.Array,
    input_shapes: Sequence[tuple],
    learning_rate: float,
    warmup_steps: int = 500,
    decay_steps: int = 2000,
) -> TrainState:
    key, params_key, dropout_key = jax.random.split(key, 3)
    init_inputs = [jnp.zeros(shape=shape) for shape in input_shapes]
    variables = model.init(
        {"params": params_key, "dropout": dropout_key},
        *init_inputs,
        train=True,
    )
    
    # Print the number of trainable parameters
    num_params = sum(param.size for param in jax.tree_util.tree_leaves(variables["params"]))
    print(f"Number of trainable parameters: {num_params:,}")
    params = variables["params"]
    batch_stats = variables["batch_stats"] if "batch_stats" in variables else {}

    optimizer = create_optimizer(learning_rate=learning_rate, warmup_steps=warmup_steps, decay_steps=decay_steps)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        key=key,
        tx=optimizer,
        batch_stats=batch_stats,
    )

    return state


@jax.jit
def eval_score(state: TrainState, val: jax.Array, time: jnp.ndarray) -> jax.Array:
    assert len(val.shape) == 1
    time = jnp.array(time)
    val_real, val_imag = val.real, val.imag
    score_real, score_imag = state.apply_fn(
        {"params": state.params},
        x_real=val_real,
        x_imag=val_imag,
        t=time,
        train=False,
    )
    return score_real + 1j * score_imag


def score_fn(state: TrainState) -> Callable:
    @jax.jit
    def score(val, time):
        result = state.apply_fn(
            {"params": state.params, "batch_stats": state.batch_stats},
            x=val[None],
            t=jnp.asarray([time]),
            train=False,
        )
        return result

    return score


def get_iterable_dataset(generator: callable, dtype: any, shape: any):
    if type(dtype) == tf.DType and type(shape) == list:
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(tf.TensorSpec(shape=shape, dtype=dtype)),
        )
    elif type(dtype) == tuple and type(shape) == list:
        assert len(dtype) == len(shape)
        signatures = tuple([tf.TensorSpec(shape=shape[i], dtype=dtype[i]) for i in range(len(dtype))])
        dataset = tf.data.Dataset.from_generator(generator, output_signature=signatures)
    else:
        raise ValueError("Invalid dtype or shape")
    iterable_dataset = iter(tfds.as_numpy(dataset))
    return iterable_dataset


def bse(xs: jnp.ndarray, Ws: jnp.ndarray) -> jnp.ndarray:
    """Batched squared error,
    xs.shape: (aux_dim, n_bases, dim)
    Ws.shape: (aux_dim, n_bases, n_bases)

    Returns: scalar
    """
    assert xs.shape[0] == Ws.shape[0]
    if xs.shape[0] == Ws.shape[0] == 1:  # real matrices
        Wx = (Ws[0] @ xs[0]).flatten()
        xWx = jnp.dot(xs[0].flatten(), Wx)
        return xWx  # scalar
    elif xs.shape[0] == Ws.shape[0] == 2:  # complex matrices
        xs = xs[0] + 1j * xs[1]
        Ws = Ws[0] + 1j * Ws[1]
        xW = jnp.matmul(xs.conj().T, Ws)
        xWx = jnp.matmul(xW, xs)
        return jnp.abs(jnp.trace(xWx))
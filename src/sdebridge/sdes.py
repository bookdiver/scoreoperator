import dataclasses
import functools
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp

from utils import invert, mult


@dataclasses.dataclass(frozen=True, eq=True)
class SDE:
    T: float
    Nt: int
    dim: int
    n_bases: int
    drift: Callable
    diffusion: Callable
    bm_size: int
    params: Any

    @property
    def bm_shape(self):
        return (self.bm_size, self.dim)

    @property
    def dt(self):
        return self.T / self.Nt

    @property
    def ts(self):
        return jnp.linspace(0, self.T, self.Nt)


@partial(jax.jit, static_argnums=0)
def cov(sde: SDE, val: jnp.ndarray, time: jnp.ndarray) -> jax.Array:
    """covariance term: sigma @ sigma^T"""
    _diffusion = sde.diffusion(val, time)
    return mult(_diffusion, _diffusion.transpose(0, 2, 1), B_conj=True)


@partial(jax.jit, static_argnums=0)
def cov_div(sde: SDE, val: jnp.ndarray, time: jnp.ndarray) -> jax.Array:
    # NOTE: Probably wrong now
    """divergence of covariance term: nabla_x (sigma @ sigma^T)"""
    jacobian_ = jnp.stack([jax.jacfwd(cov, argnums=1)(sde, val[i], time) for i in range(val.shape[0])], axis=0)
    return jnp.trace(jacobian_)


@partial(jax.jit, static_argnums=(0, 2))
def simulate_traj(sde: SDE, initial_val: jax.Array, num_batches: int, key: jax.Array):
    keys = jax.random.split(key, num_batches)
    euler = functools.partial(
        euler_maruyama,
        x0=initial_val,
        ts=sde.ts,
        drift=sde.drift,
        diffusion=sde.diffusion,
        bm_shape=sde.bm_shape,
    )
    batched_eul = jax.vmap(euler, in_axes=0)
    trajectories = batched_eul(keys)
    return trajectories


def reverse(sde: SDE, score_fun: Callable) -> SDE:
    """Drift and diffusion for backward bridge process (Z*(t)):
        dZ*(t) = {-f(T-t, Z*(t)) + Sigma(T-t, Z*(t)) s(T-t, Z*(t)) + div Sigma(T-t, Z*(t))} dt + g(T-t, Z*(t)) dW(t)

        score_fun (callable): nabla log p(x, t), either a closed form or a neural network.

    Returns:
        results: trajectories: jax.Array, (B, n_bases, d) backward bridge trajectories
    !!! n_bases.B. trajectories = [Z*(T), ..., Z*(0)], which is opposite to expected simulate_backward_bridge !!!
    """

    def drift(val: jax.Array, time: jax.Array) -> jax.Array:
        inverted_time = sde.T - time
        rev_drift_term = sde.drift(val=val, time=inverted_time)  # (aux_dim, n_bases, dim)

        cov_ = cov(sde, val=val, time=inverted_time)  # (aux_dim, n_bases, n_bases)
        score_ = jnp.squeeze(score_fun(val=val, time=inverted_time), axis=0)  # (aux_dim*n_bases*dim)
        score_term = mult(cov_, score_)  # (aux_dim, n_bases, dim)

        # NOTE: Haven't figured out how to compute this term yet
        # div_term = cov_div(sde, val=val, time=inverted_time) # (aux_dim, n_bases, dim)
        # return -rev_drift_term + score_term + div_term
        return -rev_drift_term + score_term

    def diffusion(val: jax.Array, time: jax.Array) -> jax.Array:
        inverted_time = sde.T - time
        return sde.diffusion(val=val, time=inverted_time)

    return SDE(
        T=sde.T,
        Nt=sde.Nt,
        dim=sde.dim,
        n_bases=sde.n_bases,
        drift=drift,
        diffusion=diffusion,
        bm_size=sde.bm_size,
        params=sde.params,
    )


def bridge(sde: SDE, score_fun: Callable) -> SDE:
    """Drift and diffusion for the forward bridge process (X*(t)) which is the "backward of backward":
        dX*(t) = {-f(t, X*(t)) + Sigma(t, X*(t)) [s*(t, X*(t)) - s(t, X*(t))]} dt + g(t, X*(t)) dW(t)

    Args:
        score_fun  (callable): nabla log h(x, t), either a closed form or a neural network.

    Returns:
        results: "trajectories": jax.Array, (B, n_bases, d) forward bridge trajectories (in normal order)
    """

    def drift(val: jax.Array, time: jax.Array) -> jax.Array:
        orig_drift_term = sde.drift(val=val, time=time)

        _score = jnp.squeeze(score_fun(val=val, time=time))  # (aux_dim*n_bases*dim)
        _covariance = cov(sde, val=val, time=time)
        score_term = jnp.dot(_covariance, _score)
        return orig_drift_term + score_term

    def diffusion(val: jax.Array, time: jax.Array) -> jax.Array:
        return sde.diffusion(val=val, time=time)

    return SDE(
        T=sde.T,
        Nt=sde.Nt,
        dim=sde.dim,
        n_bases=sde.n_bases,
        drift=drift,
        diffusion=diffusion,
        bm_size=sde.bm_size,
        params=sde.params,
    )


def euler_maruyama(key, x0, ts, drift, diffusion, bm_shape) -> jnp.ndarray:
    """
    Normal Euler-Maruyama solver without recording the covariance and gradient,
    used for forward simulation.
    x0.shape: (aux_dim, n_bases, dim)

    Return xs: (Nt+1, aux_dim, n_bases, dim)
    """

    def step_fun(key_and_x_and_t, dt):
        k, x, t = key_and_x_and_t
        k, subkey = jax.random.split(k, num=2)
        eps = jax.random.normal(subkey, shape=(x.shape[0],) + bm_shape)  # (aux_dim, bm_size, dim)
        diffusion_ = diffusion(x, t)  # (aux_dim, n_bases, bm_size)
        xnew = x + dt * drift(x, t) + jnp.sqrt(dt) * mult(diffusion_, eps)  # (aux_dim, n_bases, dim)
        tnew = t + dt

        return (k, xnew, tnew), xnew

    init = (key, x0, ts[0])
    _, x_all = jax.lax.scan(step_fun, xs=jnp.diff(ts), init=init)
    xs = jnp.concatenate([x0[None], x_all], axis=0)
    return xs


def brownian_sde(T, Nt, dim, n_bases, sigma) -> SDE:
    def drift(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(val)  # (aux_dim, n_bases, dim)

    def diffusion(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        a = val.shape[0]
        return jnp.tile(
            jnp.expand_dims(sigma * jnp.eye(n_bases), axis=0), reps=(a, 1, 1)
        )  # (aux_dim, n_bases, n_bases)

    return SDE(
        T=T,
        Nt=Nt,
        dim=dim,
        n_bases=n_bases,
        drift=drift,
        diffusion=diffusion,
        bm_size=n_bases,
        params=None,
    )


def trace_brownian_sde(T, Nt, dim, n_bases, alpha, power) -> SDE:
    def drift(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(val)  # (aux_dim, n_bases, dim)

    def diffusion(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        k = jnp.arange(1, n_bases + 1)
        a = val.shape[0]
        return jnp.tile(
            jnp.expand_dims(jnp.diag(alpha / k**power), axis=0), reps=(a, 1, 1)
        )  # (aux_dim, n_bases, n_bases)

    return SDE(
        T=T,
        Nt=Nt,
        dim=dim,
        n_bases=n_bases,
        drift=drift,
        diffusion=diffusion,
        bm_size=n_bases,
        params=None,
    )


def gaussian_kernel_sde(T, Nt, dim, n_bases, alpha, sigma) -> SDE:
    def drift(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(val)  # (aux_dim, n_bases, dim)

    def diffusion(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        Q_half = kernel_gaussian_Q_half(landmarks=val.squeeze(), alpha=alpha, sigma=sigma)
        a = val.shape[0]
        return jnp.tile(jnp.expand_dims(Q_half, axis=0), reps=(a, 1, 1))  # (aux_dim, n_bases, n_bases)

    return SDE(
        T=T,
        Nt=Nt,
        dim=dim,
        n_bases=n_bases,
        drift=drift,
        diffusion=diffusion,
        bm_size=n_bases,
        params=None,
    )


def gaussian_independent_kernel_sde(
    T: float,
    Nt: int,
    dim: int,
    n_bases: int,
    alpha: float,
    sigma: float,
    Ngrid: int,
    grid_range: tuple,
) -> SDE:
    def drift(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(val)  # (aux_dim, n_bases, dim)

    def diffusion(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        kernel = kernel_gaussian_independent(alpha, sigma, grid_range=grid_range, Ngrid=Ngrid)
        val = val.squeeze().reshape(-1, dim)  # (aux_dim, n_bases, dim)
        Q_half = kernel(val)  # (n_bases, n_grid**2)
        return jnp.expand_dims(Q_half, axis=0)  # (aux_dim, n_bases, bm_size)

    return SDE(
        T=T,
        Nt=Nt,
        dim=dim,
        n_bases=n_bases,
        drift=drift,
        diffusion=diffusion,
        bm_size=Ngrid**2,
        params=None,
    )


def fourier_gaussian_kernel_sde(T, Nt, dim, n_bases, alpha, sigma, n_grid, grid_range, n_pts) -> SDE:
    # if n_samples / 2 + 1 < n_bases:
    #     raise ValueError("(n_samples/2 + 1)  must be more than n_bases")

    def inverse_fourier(coefficients, Npt):
        """
        coefficients.shape: [..., 2, n_bases, dim]
        Returns array of shape [..., n_pts, dim]
        """
        assert coefficients.shape[-2] % 2 == 0
        coeffs_real = coefficients[..., 0, :, :]
        coeffs_imag = coefficients[..., 1, :, :]
        complex_coefficients = coeffs_real + 1j * coeffs_imag
        return jnp.fft.irfft(complex_coefficients, norm="forward", n=Npt, axis=-2)

    gaussian_grid_kernel = kernel_gaussian_independent(alpha, sigma, grid_range, n_grid, dim)

    def evaluate_Q(X_coeffs: jnp.ndarray) -> jnp.ndarray:
        """
        X_coeffs.shape: (aux_dim=2, n_bases, dim)
        """
        X_pts = inverse_fourier(X_coeffs, n_pts)  # (n_pts, dim)
        Q_half = gaussian_grid_kernel(X_pts)  # (n_pts, n_grid**2)
        return Q_half.reshape(n_pts, n_grid, n_grid)  # (n_pts, n_grid, n_grid)

    def drift(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(val)

    @jax.jit
    def diffusion(val: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        Q_eval = evaluate_Q(val)  # (n_pts, n_grid, n_grid)
        Q_eval = jnp.fft.rfft(Q_eval, axis=0, norm="forward")[:n_bases, ...]  # (n_bases, n_grid, n_grid)
        Q_eval = jnp.fft.ifft2(Q_eval, axes=(1, 2), norm="backward")  # (n_bases, n_grid, n_grid)

        diff = Q_eval.reshape(n_bases, n_grid**2)
        coeffs = jnp.stack([diff.real, diff.imag], axis=0)  # (2, n_bases, n_grid**2)

        return coeffs

    return SDE(T, Nt, dim, n_bases, drift, diffusion, bm_size=n_grid**2, params=None)


def kernel_gaussian_Q_half(landmarks: jnp.ndarray, alpha: float, sigma: float) -> jax.Array:
    xy_coords = landmarks.reshape(-1, 2)
    diff = xy_coords[:, jnp.newaxis, :] - xy_coords[jnp.newaxis, :, :]
    dis = jnp.sum(jnp.square(diff), axis=-1)
    Q_half = alpha * jnp.exp(-dis / sigma**2)
    return Q_half


def kernel_gaussian_2d(alpha: float, sigma: float) -> callable:
    def k(x, y):
        return alpha * jnp.exp(-0.5 * jnp.sum(jnp.square(x - y), axis=-1) / (sigma**2))

    return k


def kernel_gaussian_independent(alpha, sigma, grid_range, Ngrid, dim=2):
    grid = jnp.linspace(*grid_range, Ngrid)
    grid = jnp.stack(jnp.meshgrid(grid, grid, indexing="xy"), axis=-1)
    grid_ = grid.reshape(-1, dim)  # (n_grid**2, dim)

    gauss_kernel = kernel_gaussian_2d(alpha, sigma)
    batch_over_grid = jax.vmap(gauss_kernel, in_axes=(None, 0))
    batch_over_vals = jax.vmap(batch_over_grid, in_axes=(0, None))

    def kernel(val):
        Q_half = batch_over_vals(val, grid_)
        Q_half = Q_half.reshape(-1, Ngrid**2)  # (n_pts, n_grid**2)
        return Q_half

    return kernel
from typing import Callable, Tuple
import functools
from contextlib import contextmanager

import jax
import jax.numpy as jnp
from jax.random import PRNGKey, normal

@contextmanager
def use_cpu_backend():
    original_backend = jax.config.read("jax_platform_name")
    jax.config.update("jax_platform_name", "cpu")

    try:
        yield
    finally:
        jax.config.update("jax_platform_name", original_backend)

def cpu_run(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with jax.default_device(jax.devices("cpu")[0]):
            return func(*args, **kwargs)
    return wrapper


@functools.partial(jax.jit, )
def euler_maruyama_solver(
    rng_key: PRNGKey, 
    f: Callable[[float, jnp.ndarray], jnp.ndarray],
    g: Callable[[float, jnp.ndarray], jnp.ndarray],
    *,
    x0: jnp.ndarray,
    t0: float = 0.0,
    t1: float = 1.0,
    n_steps: int = 100,
    start: bool = False,
    noise_dim: int = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """ Euler-Maruyama solver for solving SDEs

    Args:
        rng_key (PRNGKey): random generator key.
        f (Callable[[float, jnp.ndarray], jnp.ndarray]): drift term of the SDE.
        g (Callable[[float, jnp.ndarray], jnp.ndarray]): diffusion term of the SDE.
        x0 (jnp.ndarray): initial value of the SDE.
        t0 (float, optional): starting time. Defaults to 0.0.
        t1 (float, optional): end time. Defaults to 1.0.
        n_steps (int, optional): number of solving steps. Defaults to 100.
        start (bool, optional): whether to include the initial value. Defaults to False.
        noise_dim (int, optional): dimension of the noise. Defaults to None.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: time steps and the corresponding values.
    """
    ts = jnp.linspace(t0, t1, n_steps+1, endpoint=True)         # (n_steps+1, )
    
    assert noise_dim is not None, "noise_dim should be provided."
    brownians = normal(rng_key, (n_steps, noise_dim))           # (n_steps, noise_dim)
    
    def step(x: jnp.ndarray, carry: Tuple[float, float, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ Step function for the solver

        Args:
            x (jnp.ndarray): current value.
            carry (Tuple[float, float, jnp.ndarray]): carry information (t, dt, brownian)
            
        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: doubled next value.
        """
        t, dt, brownian = carry
        x_next = x + f(t, x) * dt + jnp.dot(g(t, x), jnp.sqrt(dt) * brownian)
        return x_next, x_next
    
    _, ys = jax.lax.scan(
        f=step,
        init=x0,
        xs=jnp.hstack([ts[:-1][:, None], jnp.diff(ts)[:, None], brownians])
    )
    
    if start:
        ys = jnp.vstack([x0, ys])
        return ts, ys
    else:
        return ts[1:], ys
        

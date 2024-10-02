from __future__ import annotations
import abc
from dataclasses import dataclass, field
from typing import Dict, Tuple
from functools import partial

import jax
import jax.numpy as jnp

from ..diffusion.sde import BaseSDE

@dataclass
class SamplePath:
    name: str = field(default="")
    path: Dict[str, jnp.ndarray] = field(default_factory=dict)
    
    def __init__(self, name: str = "", **kwargs):
        self.name = name
        self.path = {}
        for key, value in kwargs.items():
            self.add(key, value)

    def __str__(self) -> str:
        info = [f"{self.name} sample path contains {self.n_batches} samples，each sample runs {self.n_steps} steps:"]
        info.extend(f"{key}.shape: {value.shape}" for key, value in self.path.items())
        return "\n ".join(info)
    
    def __getitem__(self, idx: int) -> Dict[str, jnp.ndarray]:
        if idx >= self.n_steps:
            raise IndexError(f"Index out of range: {idx} >= {self.n_steps}")
        return {key: value[:, idx] for key, value in self.path.items()}
    
    def __getattr__(self, key: str) -> jnp.ndarray:
        try:
            return self.path[key]
        except KeyError:
            raise AttributeError(f"{type(self).__name__} has no attribute '{key}'")
    
    @property
    def n_steps(self) -> int:
        return next(iter(self.path.values())).shape[1] if self.path else 0
        
    @property
    def n_batches(self) -> int:
        return next(iter(self.path.values())).shape[0] if self.path else 0
    
    def add(self, key: str, val: jnp.ndarray) -> None:
        if not isinstance(val, jnp.ndarray):
            raise ValueError(f"Only jnp.ndarray is allowed, but received {type(val)}")
        self.path[key] = val

    def copy(self) -> "SamplePath":
        return SamplePath(self.name, {k: v.copy() for k, v in self.path.items()})
    
    def tree_flatten(self):
        return jax.tree_util.tree_flatten(self.path)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(name=aux_data, **jax.tree_util.tree_unflatten(aux_data, children))

jax.tree_util.register_pytree_node(SamplePath, SamplePath.tree_flatten, SamplePath.tree_unflatten)

class Wiener:
    def __init__(self, 
                 W_shape: Tuple[int, ...],
                 T: float = 1.0,
                 dt: float = 1e-2
                 ):
        self.W_size = jnp.prod(jnp.asarray(W_shape))
        self.T = T
        self.dt = dt
    
    @property
    def ts(self) -> jnp.ndarray:
        return jnp.arange(0.0, self.T + self.dt, self.dt)
    
    @property
    def n_steps(self) -> int:
        return len(self.ts) - 1
    
    @property
    def dts(self) -> jnp.ndarray:
        return jnp.diff(self.ts)

    def _sample_step(self, rng_key: jax.Array, dt: float | None = None) -> jnp.ndarray:
        if dt is None:
            dt = self.dt
        return jax.random.normal(rng_key, shape=(self.W_size, )) * jnp.sqrt(dt)

    @partial(jax.jit, static_argnums=(0, 3))
    def sample_path(self, rng_key: jax.Array, ts: jnp.ndarray | None = None, n_batches: int = 1) -> SamplePath:
        if ts is None:
            dts = self.dts
            n_steps = self.n_steps
        else:
            dts = jnp.diff(ts)
            n_steps = len(dts)
        
        subkeys = jax.random.split(rng_key, n_steps * n_batches).reshape((n_batches, n_steps, -1))
        dWs = jax.vmap(
            jax.vmap(
                self._sample_step,
                in_axes=(0, 0)
            ),
            in_axes=(0, None)
        )(subkeys, dts)
        
        return SamplePath(
            name="Wiener process",
            ts=ts,
            xs=dWs
        )
        
class SDESolver(abc.ABC):
    sde: BaseSDE
    wiener: Wiener

    def __init__(self, sde: BaseSDE, wiener: Wiener):
        self.sde = sde
        self.wiener = wiener
        self._ts = wiener.ts
        self._dts = jnp.diff(self._ts)
        
    @abc.abstractmethod
    def _solve_step(self, x: jnp.ndarray, t: jnp.ndarray, dt: float, dW: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        pass

    @partial(jax.jit, static_argnums=(0, 3))
    def solve(
        self,
        rng_key: jax.Array,
        x0: jnp.ndarray,
        n_batches: int = 1,
        *args, 
        **kwargs
    ) -> SamplePath:
        
        def scan_fn(carry: Tuple[jnp.ndarray, ...], val: Tuple[jnp.ndarray, float, jnp.ndarray], *args, **kwargs) -> Tuple[Tuple[jnp.ndarray, ...], jnp.ndarray]:
            x, *_ = carry
            t, dt, dW = val
            x_next = self._solve_step(x, t, dt, dW, *args, **kwargs)
            return (x_next,), x_next
        
        dWs = self.wiener.sample_path(
            rng_key,
            ts=self._ts,
            n_batches=n_batches
        ).xs
        
        _, xs = jax.vmap(
            lambda dW: jax.lax.scan(
                scan_fn,
                init=(x0.flatten(),),
                xs=(self._ts[:-1], self._dts, dW)
            ),
            in_axes=0
        )(dWs)
        
        return SamplePath(xs=xs, ts=self._ts[1:])
    
class EulerMaruyama(SDESolver):
    
    def _solve_step(self, x: jnp.ndarray, t: jnp.ndarray, dt: float, dW: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        x_next = x + self.sde.f(t, x, *args, **kwargs) * dt + self.sde.g(t, x, *args, **kwargs) @ dW
        return x_next
    
    
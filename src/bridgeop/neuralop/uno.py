from flax import linen as nn
from typing import Tuple, Union

from .blocks import *

class CTUNO(nn.Module):
    """ U-Net shaped time-dependent neural operator"""
    do_dim: int
    in_co_dim: int
    out_co_dim: int
    lifting_dim: int
    co_dims_fmults: Tuple[int, ...]
    grid_scaling_fmults: Tuple[float, ...]
    n_modes_per_layer: Tuple[Tuple[int, ...],...]
    norm: str = "instance"
    act: str  = "relu"
    use_freq_mod: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        assert len(self.co_dims_fmults) == len(self.n_modes_per_layer) == len(self.grid_scaling_fmults)
        t_emb_dim = 4 * self.lifting_dim
        co_dims_fmults = (1,) + self.co_dims_fmults

        t_emb = TimeEmbedding(
            t_emb_dim,
        )(t)

        x = nn.Dense(
            self.lifting_dim,
        )(x)


        downs = []
        for idx_layer in range(len(self.co_dims_fmults)):
            in_co_dim_fmult = co_dims_fmults[idx_layer]
            out_co_dim_fmult = co_dims_fmults[idx_layer+1]
            n_modes = self.n_modes_per_layer[idx_layer]
            x = CTUNOBlock(
                do_dim=self.do_dim,
                in_co_dim=int(self.lifting_dim * in_co_dim_fmult),
                out_co_dim=int(self.lifting_dim * out_co_dim_fmult),
                t_emb_dim=t_emb_dim,
                n_modes=n_modes,
                out_grid_scaling=self.grid_scaling_fmults[idx_layer],
                norm=self.norm,
                act=self.act,
                use_freq_mod=self.use_freq_mod
            )(x, t_emb, train)
            downs.append(x)

        x = CTUNOBlock(
            do_dim=self.do_dim,
            in_co_dim=self.lifting_dim * self.co_dims_fmults[-1],
            out_co_dim=self.lifting_dim * self.co_dims_fmults[-1],
            t_emb_dim=t_emb_dim,
            n_modes=self.n_modes_per_layer[-1],
            out_grid_scaling=1.0,
            norm=self.norm,
            act=self.act,
            use_freq_mod=self.use_freq_mod
        )(x, t_emb, train)

        for idx_layer in range(len(self.co_dims_fmults)-1, 0, -1):
            in_co_dim_fmult = co_dims_fmults[idx_layer+1]
            out_co_dim_fmult = co_dims_fmults[idx_layer] 
            n_modes = self.n_modes_per_layer[idx_layer]
            down = downs[idx_layer]
            x = jnp.concatenate([x, down], axis=-1)
            x = CTUNOBlock(
                do_dim=self.do_dim,
                in_co_dim=int(self.lifting_dim * in_co_dim_fmult * 2),
                out_co_dim=int(self.lifting_dim * out_co_dim_fmult),
                t_emb_dim=t_emb_dim,
                n_modes=n_modes,
                out_grid_scaling=1.0 / self.grid_scaling_fmults[idx_layer],
                norm=self.norm,
                act=self.act,
                use_freq_mod=self.use_freq_mod
            )(x, t_emb, train)
        
        x = nn.Dense(
            self.out_co_dim,
        )(x)

        return x
        




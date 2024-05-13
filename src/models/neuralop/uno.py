from dataclasses import field
from flax import linen as nn

from .blocks import *

class CTUNO1D(nn.Module):
    """ U-Net shaped time-dependent neural operator"""
    out_co_dim: int
    lifting_dim: int
    co_dims_fmults: tuple
    n_modes_per_layer: tuple
    norm: str = "instance"
    act: str  = "relu"

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        t_emb_dim = 4 * self.lifting_dim
        _, in_grid_sz, _ = x.shape
        co_dims_fmults = (1,) + self.co_dims_fmults

        t_emb = TimeEmbedding(
            t_emb_dim,
        )(t)

        x = nn.Dense(
            self.lifting_dim,
        )(x)

        out_grid_sz_fmults = [1. / dim_fmult for dim_fmult in co_dims_fmults]

        downs = []
        for idx_layer in range(len(self.co_dims_fmults)):
            in_co_dim_fmult = co_dims_fmults[idx_layer]
            out_co_dim_fmult = co_dims_fmults[idx_layer+1]
            out_grid_sz = int(out_grid_sz_fmults[idx_layer+1] * in_grid_sz)
            n_modes = self.n_modes_per_layer[idx_layer]
            x = CTUNOBlock1D(
                in_co_dim=int(self.lifting_dim * in_co_dim_fmult),
                out_co_dim=int(self.lifting_dim * out_co_dim_fmult),
                t_emb_dim=t_emb_dim,
                n_modes=n_modes,
                out_grid_sz=out_grid_sz,
                norm=self.norm,
                act=self.act
            )(x, t_emb, train)
            downs.append(x)

        x = CTUNOBlock1D(
            in_co_dim=self.lifting_dim * self.co_dims_fmults[-1],
            out_co_dim=self.lifting_dim * self.co_dims_fmults[-1],
            t_emb_dim=t_emb_dim,
            n_modes=self.n_modes_per_layer[-1],
            out_grid_sz=int(out_grid_sz_fmults[-1] * in_grid_sz),
            norm=self.norm,
            act=self.act
        )(x, t_emb, train)

        for idx_layer in range(1, len(self.co_dims_fmults)+1):
            in_co_dim_fmult = co_dims_fmults[-idx_layer]
            out_co_dim_fmult = co_dims_fmults[-(idx_layer+1)] 
            out_grid_sz = int(out_grid_sz_fmults[-(idx_layer+1)] * in_grid_sz)
            n_modes = self.n_modes_per_layer[-idx_layer]
            down = downs[-idx_layer]
            x = jnp.concatenate([x, down], axis=-1)
            x = CTUNOBlock1D(
                in_co_dim=int(self.lifting_dim * in_co_dim_fmult * 2),
                out_co_dim=int(self.lifting_dim * out_co_dim_fmult),
                t_emb_dim=t_emb_dim,
                n_modes=n_modes,
                out_grid_sz=out_grid_sz,
                norm=self.norm,
                act=self.act
            )(x, t_emb, train)
        
        x = nn.Dense(
            self.out_co_dim,
        )(x)

        return x


class CTUNO2D(nn.Module):
    """ U-Net shaped time-dependent neural operator"""
    out_co_dim: int
    lifting_dim: int
    co_dims_fmults: tuple
    n_modes_per_layer: tuple
    norm: str = "instance"
    act: str  = "relu"

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        t_emb_dim = 4 * self.lifting_dim
        _, in_grid_sz, _, _ = x.shape
        co_dims_fmults = (1,) + self.co_dims_fmults

        t_emb = TimeEmbedding(
            t_emb_dim,
        )(t)

        x = nn.Conv(
            features=self.lifting_dim,
            kernel_size=(1, 1),
            padding="VALID"
        )(x)

        out_grid_sz_fmults = [1. / dim_fmult for dim_fmult in co_dims_fmults]

        downs = []
        for idx_layer in range(len(self.co_dims_fmults)):
            in_co_dim_fmult = co_dims_fmults[idx_layer]
            out_co_dim_fmult = co_dims_fmults[idx_layer+1]
            out_grid_sz = int(out_grid_sz_fmults[idx_layer+1] * in_grid_sz)
            n_modes = self.n_modes_per_layer[idx_layer]
            x = CTUNOBlock2D(
                in_co_dim=int(self.lifting_dim * in_co_dim_fmult),
                out_co_dim=int(self.lifting_dim * out_co_dim_fmult),
                t_emb_dim=t_emb_dim,
                n_modes=n_modes,
                out_grid_sz=out_grid_sz,
                norm=self.norm,
                act=self.act
            )(x, t_emb, train)
            downs.append(x)

        x = CTUNOBlock2D(
            in_co_dim=self.lifting_dim * self.co_dims_fmults[-1],
            out_co_dim=self.lifting_dim * self.co_dims_fmults[-1],
            t_emb_dim=t_emb_dim,
            n_modes=self.n_modes_per_layer[-1],
            out_grid_sz=int(out_grid_sz_fmults[-1] * in_grid_sz),
            norm=self.norm,
            act=self.act
        )(x, t_emb, train)

        for idx_layer in range(1, len(self.co_dims_fmults)+1):
            in_co_dim_fmult = co_dims_fmults[-idx_layer]
            out_co_dim_fmult = co_dims_fmults[-(idx_layer+1)] 
            out_grid_sz = int(out_grid_sz_fmults[-(idx_layer+1)] * in_grid_sz)
            n_modes = self.n_modes_per_layer[-idx_layer]
            down = downs[-idx_layer]
            x = jnp.concatenate([x, down], axis=-1)
            x = CTUNOBlock2D(
                in_co_dim=int(self.lifting_dim * in_co_dim_fmult * 2),
                out_co_dim=int(self.lifting_dim * out_co_dim_fmult),
                t_emb_dim=t_emb_dim,
                n_modes=n_modes,
                out_grid_sz=out_grid_sz,
                norm=self.norm,
                act=self.act
            )(x, t_emb, train)
        
        x = nn.Conv(
            features=self.out_co_dim,
            kernel_size=(1, 1),
            padding="VALID"
        )(x)

        return x
        




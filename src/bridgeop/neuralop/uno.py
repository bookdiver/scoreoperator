from flax import linen as nn

from .blocks import *

class CTUNO(nn.Module):
    """ U-Net shaped time-dependent neural operator"""
    in_co_dim: int
    out_co_dim: int
    lifting_dim: int
    co_dims_fmults: tuple
    n_modes_per_layer: tuple
    norm: str = "instance"
    act: str  = "relu"
    use_freq_mod: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        t_emb_dim = 4 * self.lifting_dim
        n_batches = x.shape[0]
        x = x.reshape(n_batches, -1, self.in_co_dim)
        in_grid_sz = x.shape[1]
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
            x = CTUNOBlock(
                in_co_dim=int(self.lifting_dim * in_co_dim_fmult),
                out_co_dim=int(self.lifting_dim * out_co_dim_fmult),
                t_emb_dim=t_emb_dim,
                n_modes=n_modes,
                out_grid_sz=out_grid_sz,
                norm=self.norm,
                act=self.act,
                use_freq_mod=self.use_freq_mod
            )(x, t_emb, train)
            downs.append(x)

        x = CTUNOBlock(
            in_co_dim=self.lifting_dim * self.co_dims_fmults[-1],
            out_co_dim=self.lifting_dim * self.co_dims_fmults[-1],
            t_emb_dim=t_emb_dim,
            n_modes=self.n_modes_per_layer[-1],
            out_grid_sz=int(out_grid_sz_fmults[-1] * in_grid_sz),
            norm=self.norm,
            act=self.act,
            use_freq_mod=self.use_freq_mod
        )(x, t_emb, train)

        for idx_layer in range(1, len(self.co_dims_fmults)+1):
            in_co_dim_fmult = co_dims_fmults[-idx_layer]
            out_co_dim_fmult = co_dims_fmults[-(idx_layer+1)] 
            out_grid_sz = int(out_grid_sz_fmults[-(idx_layer+1)] * in_grid_sz)
            n_modes = self.n_modes_per_layer[-idx_layer]
            down = downs[-idx_layer]
            x = jnp.concatenate([x, down], axis=-1)
            x = CTUNOBlock(
                in_co_dim=int(self.lifting_dim * in_co_dim_fmult * 2),
                out_co_dim=int(self.lifting_dim * out_co_dim_fmult),
                t_emb_dim=t_emb_dim,
                n_modes=n_modes,
                out_grid_sz=out_grid_sz,
                norm=self.norm,
                act=self.act,
                use_freq_mod=self.use_freq_mod
            )(x, t_emb, train)
        
        x = nn.Dense(
            self.out_co_dim,
        )(x)

        x = x.reshape(n_batches, -1)

        return x
        




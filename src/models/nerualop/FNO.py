from typing import List
from flax import linen as nn

from .blocks import *
    
class TimeDependentFNO1D(nn.Module):
    """ Time dependent Fourier Neural Operator for mapping: (v: R -> R^{input_dim}) -> (u: R -> R^{output_dim})"""
    output_dim: int
    lifting_dims: list
    max_n_modes: list
    activation: str
    time_incrop_method: str
    time_embedding_dim: int

    @nn.compact
    def __call__(self, x, t, train):
        """ x shape: (batch, n_samples, input_dim), 
            t shape: (batch, ) 

            output shape: (batch, n_samples, output_dim)
        """
        if self.time_incrop_method == 'resnet':
            block_cls = ResidualFourierBlock1D
        elif self.time_incrop_method == 'time_modulated':
            block_cls = TimeModulatedFourierBlock1D
        else:
            raise ValueError(f"Invalid time_incrop_method: {self.time_incrop_method}")
        
        t_emb = TimeEmbedding(
            self.time_embedding_dim,
        )(t)
        
        x = nn.Dense(
            self.lifting_dims[0],
        )(x)
        
        for i in range(len(self.lifting_dims)-1):
            x = block_cls(
                self.lifting_dims[i],
                self.lifting_dims[i+1],
                self.time_embedding_dim,
                n_modes=self.max_n_modes[i],
                activation=self.activation,
            )(x, t_emb, train)

        x = nn.Dense(
            self.output_dim,
        )(x)
        
        return x

class UNO1D(nn.Module):
    """ U-Net shaped time-dependent neural operator"""
    output_dim: int
    lifting_dim: int
    n_modes_fmult: float = 1.0
    dims_fmults: List[int] = [1, 2, 4, 4]
    activation: str  = "relu"

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        time_embedding_dim = 4 * self.lifting_dim

        t_emb = TimeEmbedding(
            embedding_dim=time_embedding_dim,
        )(t)

        x_lifted = nn.Dense(
            self.lifting_dim,
        )(x)

        x_down1 = TimeModulatedFourierBlock1D(
            input_dim=self.lifting_dim,
            output_dim=self.lifting_dim * self.dims_fmults[0],
            encoding_dim=time_embedding_dim,
            n_modes=int(self.n_modes_fmult * self.lifting_dim),
            activation=self.activation,
        )(x_lifted, t_emb, train)

        x_down2 = TimeModulatedFourierBlock1D(
            input_dim=self.lifting_dim * self.dims_fmults[0],
            output_dim=self.lifting_dim * self.dims_fmults[1],
            encoding_dim=time_embedding_dim,
            n_modes=int(self.n_modes_fmult * self.lifting_dim * self.dims_fmults[0]),
            activation=self.activation,
        )(x_down1, t_emb, train)

        x_down3 = TimeModulatedFourierBlock1D(
            input_dim=self.lifting_dim * self.dims_fmults[1],
            output_dim=self.lifting_dim * self.dims_fmults[2],
            encoding_dim=time_embedding_dim,
            n_modes=int(self.n_modes_fmult * self.lifting_dim * self.dims_fmults[1]),
            activation=self.activation,
        )(x_down2, t_emb, train)

        x_down4 = TimeModulatedFourierBlock1D(
            input_dim=self.lifting_dim * self.dims_fmults[2],
            output_dim=self.lifting_dim * self.dims_fmults[3],
            encoding_dim=time_embedding_dim,
            n_modes=int(self.n_modes_fmult * self.lifting_dim * self.dims_fmults[2]),
            activation=self.activation,
        )(x_down3, t_emb, train)

        




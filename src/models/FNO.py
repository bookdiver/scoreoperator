from flax import linen as nn

from .blocks import *

class FNO1D(nn.Module):
    """ Time independent Fourier Neural Operator for mapping: (v: R -> R^{input_dim}) -> (u: R -> R^{output_dim})"""
    output_dim: int
    lifting_dims: list
    max_n_modes: list
    activation: nn.activation

    def setup(self):
        self.lifting_layer = nn.Dense(self.lifting_dims[0])
        self.fourier_blocks = [FourierBlock1D(
            self.lifting_dims[i],
            self.lifting_dims[i+1],
            n_modes=self.max_n_modes[i],
            activation=self.activation
        ) for i in range(len(self.lifting_dims)-1)]
        self.projection_layer = nn.Dense(self.output_dim)
    
    def __call__(self, x):
        """ x shape: (batch, n_samples, input_dim) """
        x = self.lifting_layer(x)
        for layer in self.fourier_blocks:
            x = layer(x)
        return self.projection_layer(x)
    
class TimeDependentFNO1D(nn.Module):
    """ Time dependent Fourier Neural Operator for mapping: (v: R -> R^{input_dim}) -> (u: R -> R^{output_dim})"""
    output_dim: int
    lifting_dims: list
    max_n_modes: list
    activation: nn.activation
    time_incrop_method: str
    time_encoding_dim: int

    def setup(self):
        self.time_encoding = TimeEncoding(self.time_encoding_dim)
        self.lifting_layer = nn.Dense(self.lifting_dims[0])

        if self.time_incrop_method == 'resnet':
            block_cls = ResidualFourierBlock1D
        elif self.time_incrop_method == 'time_modulated':
            block_cls = TMFourierBlock1D
        else:
            raise ValueError(f"Invalid time_incrop_method: {self.time_incrop_method}")

        self.fourier_blocks = [
            block_cls(
                self.lifting_dims[i],
                self.lifting_dims[i+1],
                self.time_encoding_dim,
                n_modes=self.max_n_modes[i],
                activation=self.activation
            ) for i in range(len(self.lifting_dims)-1)
        ]
        self.projection_layer = nn.Dense(self.output_dim)

    def __call__(self, x, t):
        """ x shape: (batch, n_samples, input_dim), 
            t shape: (batch, ) 
        """
        t_emb = self.time_encoding(t)
        x = self.lifting_layer(x)
        for block in self.fourier_blocks:
            x = block(x, t_emb)
        return self.projection_layer(x)

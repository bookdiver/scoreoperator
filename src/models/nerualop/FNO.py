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

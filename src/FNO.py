import jax.numpy as jnp
from flax import linen as nn

class SpectralConv1D(nn.Module):
    input_dim: int
    output_dim: int
    n_modes: int

    def setup(self):
        weights_shape = (self.n_modes//2+1, self.input_dim, self.output_dim)
        self.weights = self.param(
            'weights',
            nn.initializers.xavier_normal(),
            weights_shape
        )
    
    def __call__(self, x):
        x_ft = jnp.fft.rfft(x, axis=-2)
        out_ft = jnp.zeros((x.shape[-2]//2+1, self.output_dim), dtype=jnp.complex64)
        x_ft_transf = jnp.einsum('ni,nio->no', x_ft[:self.n_modes//2+1, :], self.weights)
        out_ft = out_ft.at[:self.n_modes//2+1, :].set(x_ft_transf)
        out = jnp.fft.irfft(out_ft, axis=-2)
        return out
    
class FourierLayer1D(nn.Module):
    input_dim: int
    output_dim: int
    n_modes: int
    activation: nn.activation = nn.relu

    def setup(self):
        self.spectral_conv = SpectralConv1D(
            self.input_dim,
            self.output_dim,
            self.n_modes,
        )
        self.residual_transf = nn.Dense(
            self.output_dim,
            kernel_init=nn.initializers.xavier_normal(),
        )
    
    def __call__(self, x):
        return self.activation(self.spectral_conv(x) + self.residual_transf(x))

class FNO1D(nn.Module):
    output_dim: int
    lifting_dims: list
    max_n_modes: list
    activation: nn.activation = nn.relu

    def setup(self):
        self.lifting_layer = nn.Dense(self.lifting_dims[0])
        self.fourier_layers = [FourierLayer1D(
            self.lifting_dims[i],
            self.lifting_dims[i+1],
            n_modes=self.max_n_modes[i],
            activation=self.activation
        ) for i in range(len(self.lifting_dims)-1)]
        self.projection_layer = nn.Dense(self.output_dim)
    
    def __call__(self, x):
        x = self.lifting_layer(x)
        for layer in self.fourier_layers:
            x = layer(x)
        return self.projection_layer(x)

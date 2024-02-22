import jax
import jax.numpy as jnp
from flax import linen as nn

class TimeEncoding(nn.Module):
    encoding_dim: int
    scaling: float = 100.0
    
    def setup(self):
        self.div_term = jnp.exp(jnp.arange(0, self.encoding_dim, 2) * -(jnp.log(10000.0) / self.encoding_dim))

    def __call__(self, t):
        pe = jnp.empty((len(t), self.encoding_dim))
        factor = self.scaling * jnp.einsum('i,j->ij', t, self.div_term)
        pe = pe.at[:, 0::2].set(jnp.sin(factor))
        pe = pe.at[:, 1::2].set(jnp.cos(factor))
        return pe

class TMSpectralConv1D(nn.Module):
    input_dim: int
    output_dim: int
    encoding_dim: int
    n_modes: int

    def setup(self):
        weights_r_shape = (self.n_modes//2+1, self.input_dim, self.output_dim)
        self.weights_r = self.param(
            'weights_r',
            nn.initializers.xavier_normal(),
            weights_r_shape
        )
        self.t_dense = nn.Dense(self.n_modes//2+1, use_bias=False, param_dtype=jnp.complex64)

    def __call__(self, x, t_emb):
        print(x.shape, t_emb.shape)
        x_ft = jnp.fft.rfft(x, axis=-2)
        out_ft = jnp.zeros((*x.shape[:-2], x.shape[-2]//2+1, self.output_dim), dtype=jnp.complex64)
        t_emb_transf = self.t_dense(t_emb)
        weights = jnp.einsum('...n,nio->...nio', t_emb_transf, self.weights_r)
        x_ft_transf = jnp.einsum('...ni,...nio->...no', x_ft[..., :self.n_modes//2+1, :], weights)
        out_ft = out_ft.at[..., :self.n_modes//2+1, :].set(x_ft_transf)
        out = jnp.fft.irfft(out_ft, axis=-2)
        return out
    
class TMFourierLayer1D(nn.Module):
    input_dim: int
    output_dim: int
    encoding_dim: int
    n_modes: int
    activation: nn.activation = nn.relu

    def setup(self):
        self.spectral_conv = TMSpectralConv1D(
            self.input_dim,
            self.output_dim,
            self.encoding_dim,
            self.n_modes,
        )
        self.residual_transf = nn.Dense(
            self.output_dim,
            kernel_init=nn.initializers.xavier_normal(),
        )
        self.t_dense = nn.Dense(
            self.input_dim,
            use_bias=False,
            kernel_init=nn.initializers.xavier_normal(),
        )
    
    def __call__(self, x, t_emb):
        t_emb = self.t_dense(t_emb)
        x_res = jnp.einsum('...d,...nd->...nd', t_emb, x)
        return self.activation(self.spectral_conv(x, t_emb) + self.residual_transf(x_res))

class TMFNO1D(nn.Module):
    output_dim: int
    lifting_dims: list
    max_n_modes: list
    encoding_dim: int
    activation: nn.activation = nn.relu

    def setup(self):
        self.time_encoding = TimeEncoding(self.encoding_dim)
        self.lifting_layer = nn.Dense(self.lifting_dims[0])
        self.fourier_layers = [
            TMFourierLayer1D(
                self.lifting_dims[i],
                self.lifting_dims[i+1],
                self.encoding_dim,
                n_modes=self.max_n_modes[i],
                activation=self.activation
            ) for i in range(len(self.lifting_dims)-1)
        ]
        self.projection_layer = nn.Dense(self.output_dim)

    def __call__(self, x, t):
        t_emb = self.time_encoding(t)
        x = self.lifting_layer(x)
        for layer in self.fourier_layers:
            x = layer(x, t_emb)
        return self.projection_layer(x)
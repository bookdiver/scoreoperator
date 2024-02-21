import jax
import jax.numpy as jnp
from flax import linen as nn
from einops import repeat

def position_encoding(t, encoding_dim):
    t = jnp.asarray(t) if not isinstance(t, jnp.ndarray) else t
    t = repeat(t, '... -> ... 1')
    pe = jnp.empty((t.shape[0], encoding_dim))
    div_term = jnp.exp(jnp.arange(0, encoding_dim, 2) * -(jnp.log(10000.0) / encoding_dim))
    pe = pe.at[:, 0::2].set(jnp.sin(t * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(t * div_term))
    return pe


class TMSpectralConv1D(nn.Module):
    input_dim: int
    output_dim: int
    encoding_dim: int
    n_modes: int

    def setup(self):
        weights_r_shape = (self.n_modes//2+1, self.input_dim, self.output_dim)
        weights_a_shape = (self.n_modes//2+1, self.encoding_dim)
        self.weights_r = self.param(
            'weights_r',
            nn.initializers.xavier_normal(),
            weights_r_shape
        )
        self.weights_a = self.param(
            'weights_a',
            nn.initializers.xavier_normal(),
            weights_a_shape
        )

    def __call__(self, x, t_emb):
        print(x.shape, t_emb.shape)
        x_ft = jnp.fft.rfft(x, axis=-2)
        out_ft = jnp.zeros((x.shape[-2]//2+1, self.output_dim), dtype=jnp.complex64)
        t_emb_transf = jnp.einsum('c,nc->n', t_emb, self.weights_a)
        weights = jnp.einsum('n,nio->nio', t_emb_transf, self.weights_r)
        x_ft_transf = jnp.einsum('ni,nio->no', x_ft[:self.n_modes//2+1, :], weights)
        out_ft = out_ft.at[:self.n_modes//2+1, :].set(x_ft_transf)
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
        x_res = self.t_dense(t_emb) * x
        return self.activation(self.spectral_conv(x, t_emb) + self.residual_transf(x_res))

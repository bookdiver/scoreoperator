import jax.numpy as jnp
from flax import linen as nn

def get_activation_fn(activation_str):
    if activation_str.lower() == 'relu':
        return nn.relu
    elif activation_str.lower() == 'tanh':
        return nn.tanh
    elif activation_str.lower() == 'silu':
        return nn.silu
    elif activation_str.lower() == 'gelu':
        return nn.gelu
    elif activation_str.lower() == 'leaky_relu':
        return nn.leaky_relu
    elif activation_str.lower() == 'elu':
        return nn.elu
    else:
        raise ValueError(f"Unknown activation function: {activation_str}")

def normal_initializer(n_modes):
    return nn.initializers.normal(stddev=jnp.sqrt(2. / n_modes))

### Fourier Layers ###
    
class SpectralConv1D(nn.Module):
    """ Integral kernel operator for mapping functions (u: R -> R^{input_dim}) to functions (v: R -> R^{output_dim}) """
    input_dim: int
    output_dim: int
    n_modes: int
    fft_norm: str = "forward"
    bias: bool = True

    @nn.compact
    def __call__(self, x):
        """ x shape: (batch, n_samples, input_dim) 

            output shape: (batch, n_samples, output_dim)
        """
        weights_shape = (self.n_modes//2+1, self.input_dim, self.output_dim)
        weights_real = self.param(
            'weights(real)',
            normal_initializer(self.n_modes),
            weights_shape
        )
        weights_imag = self.param(
            'weights(imag)',
            normal_initializer(self.n_modes),
            weights_shape
        )
        weights = weights_real + 1j*weights_imag

        if self.bias:
            bias = self.param(
                'bias',
                nn.initializers.zeros,
                (1, 1, self.output_dim)
            )

        x_ft = jnp.fft.rfft(x, axis=-2, norm=self.fft_norm)

        out_ft = jnp.zeros((*x.shape[:-2], x.shape[-2]//2+1, self.output_dim), dtype=jnp.complex64)
        x_ft_transf = jnp.einsum('...ni,nio->...no', x_ft[..., :self.n_modes//2+1, :], weights)
        out_ft = out_ft.at[..., :self.n_modes//2+1, :].set(x_ft_transf)

        out = jnp.fft.irfft(out_ft, axis=-2, norm=self.fft_norm)

        if self.bias:
            out = out + bias

        return out
    
    
class TimeModulatedSpectralConv1D(nn.Module):
    """ Time modulated integral kernel operator proposed by ``Learning PDE Solution Operator for Continuous Modelling of Time-Series`` """
    input_dim: int
    output_dim: int
    encoding_dim: int
    n_modes: int
    fft_norm: str = "forward"
    bias = True

    @nn.compact
    def __call__(self, x, t_emb):
        """ x shape: (batch, n_samples, input_dim),
            t_emb shape: (batch, encoding_dim)

            output shape: (batch, n_samples, output_dim)
        """
        weights_shape = (self.n_modes//2+1, self.input_dim, self.output_dim)
        weights_real = self.param(
            'weights(real)',
            normal_initializer(self.n_modes),
            weights_shape,
        )
        weights_imag = self.param(
            'weights(imag)',
            normal_initializer(self.n_modes),
            weights_shape,
        )
        weights = weights_real + 1j*weights_imag

        if self.bias:
            bias = self.param(
                'bias',
                nn.initializers.zeros,
                (1, 1, self.output_dim),
            )

        x_ft = jnp.fft.rfft(x, axis=-2, norm=self.fft_norm)

        out_ft = jnp.zeros((*x.shape[:-2], x.shape[-2]//2+1, self.output_dim), dtype=jnp.complex64)

        t_emb_transf_real = nn.Dense(
            self.n_modes//2+1,
            use_bias=False,
        )(t_emb)
        t_emb_transf_imag = nn.Dense(
            self.n_modes//2+1,
            use_bias=False,
        )(t_emb)
        t_emb_transf = t_emb_transf_real + 1j*t_emb_transf_imag

        weights = jnp.einsum('...n,nio->...nio', t_emb_transf, weights)
        x_ft_transf = jnp.einsum('...ni,...nio->...no', x_ft[..., :self.n_modes//2+1, :], weights)
        out_ft = out_ft.at[..., :self.n_modes//2+1, :].set(x_ft_transf)

        out = jnp.fft.irfft(out_ft, axis=-2, norm=self.fft_norm)

        if self.bias:
            out = out + bias

        return out


### FNO Blocks ###
    
class ResidualFourierBlock1D(nn.Module):
    """ Modified Fourier block to incorporate time step embedding, used by ``Score-based Diffusion Models in Function Space paper`` """
    input_dim: int
    output_dim: int
    encoding_dim: int
    n_modes: int
    activation: str
    
    @nn.compact
    def __call__(self, x, t_emb, train):
        """ x shape: (batch, n_samples, input_dim),
            t_emb shape: (batch, encoding_dim)

            output shape: (batch, n_samples, output_dim)
        """
        x_res = SpectralConv1D(
            self.input_dim,
            self.output_dim,
            self.n_modes,
        )(x)
        x_res = nn.BatchNorm(use_running_average=not train)(x_res)
        x_res = get_activation_fn(self.activation)(x_res)

        t_embs = TimeEmbeddingMLP(
            2 * self.output_dim,
            self.output_dim,
        )(t_emb)
        w, b = jnp.split(t_embs[:, None, :], 2, axis=-1)

        x_res = x_res * (w + 1.0) + b
        x_res = get_activation_fn(self.activation)(x_res)

        x_res = SpectralConv1D(
            self.output_dim,
            self.output_dim,
            self.n_modes,
        )(x_res)

        x_jump = nn.Dense(
            self.output_dim,
        )(x)

        return x_res + x_jump
    

class TimeModulatedFourierBlock1D(nn.Module):
    input_dim: int
    output_dim: int
    encoding_dim: int
    n_modes: int
    activation: str
    
    @nn.compact
    def __call__(self, x, t_emb, train):
        """ x shape: (batch, n_samples, input_dim),
            t_emb shape: (batch, encoding_dim)

            output shape: (batch, n_samples, output_dim)
        """
        t_emb = TimeEmbeddingMLP(
            self.input_dim,
            self.input_dim,
        )(t_emb)
        x_res = jnp.einsum('...d,...nd->...nd', t_emb, x)
        x_spec_out = TimeModulatedSpectralConv1D(
            self.input_dim,
            self.output_dim,
            self.encoding_dim,
            self.n_modes,
        )(x, t_emb)
        x_res_out = nn.Dense(
            self.output_dim,
        )(x_res)
        x_out = x_spec_out + x_res_out
        x_out = nn.BatchNorm(use_running_average=not train)(x_out)
        return get_activation_fn(self.activation)(x_out)
    
    
class TimeEmbedding(nn.Module):
    """ Sinusoidal time step embedding """
    embedding_dim: int
    scaling: float = 100.0
    max_period: float = 10000.0

    @nn.compact
    def __call__(self, t):
        """ t shape: (batch,) """
        pe = jnp.empty((len(t), self.embedding_dim))
        factor = self.scaling * jnp.einsum('i,j->ij', 
                                           t, 
                                           jnp.exp(jnp.arange(0, self.embedding_dim, 2) 
                                                   * -(jnp.log(self.max_period) / self.embedding_dim)))
        pe = pe.at[:, 0::2].set(jnp.sin(factor))
        pe = pe.at[:, 1::2].set(jnp.cos(factor))
        return pe
    
class TimeEmbeddingMLP(nn.Module):
    """ MLP for transformed fixed size time embedding into variable size"""
    output_dim: int
    hidden_dim: int
    
    @nn.compact
    def __call__(self, t_emb):
        """ t_emb shape: (batch, embedding_dim)

            output shape: (batch, output_dim)
        """
        t_emb = nn.Dense(
            self.hidden_dim,
        )(t_emb)

        t_emb = nn.swish(t_emb)

        t_emb = nn.Dense(
            self.output_dim,
        )(t_emb)

        return t_emb
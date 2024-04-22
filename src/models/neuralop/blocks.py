import jax
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

def normal_initializer(input_co_dim: int):
    return nn.initializers.normal(stddev=jnp.sqrt(1.0/(2.0*input_co_dim)))

### Fourier Layers ###
    
class SpectralConv1D(nn.Module):
    """ Integral kernel operator for mapping functions (u: R -> R^{input_dim}) to functions (v: R -> R^{output_dim}) """
    in_co_dim: int
    out_co_dim: int
    n_modes: int
    out_grid_sz: int = None
    fft_norm: str = "forward"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """ x shape: (batch, in_grid_sz, in_co_dim) 

            output shape: (batch, out_grid_sz, out_co_dim)
        """
        b, in_grid_sz, _ = x.shape
        out_grid_sz = in_grid_sz if self.out_grid_sz is None else self.out_grid_sz
        weights_shape = (self.n_modes//2+1, self.in_co_dim, self.out_co_dim)
        weights_real = self.param(
            'weights(real)',
            normal_initializer(self.in_co_dim),
            weights_shape
        )
        weights_imag = self.param(
            'weights(imag)',
            normal_initializer(self.in_co_dim),
            weights_shape
        )
        weights = weights_real + 1j*weights_imag

        x_ft = jnp.fft.rfft(x, axis=-2, norm=self.fft_norm)

        out_ft = jnp.zeros((b, in_grid_sz//2+1, self.out_co_dim), dtype=jnp.complex64)
        x_ft = jnp.einsum('bni,nio->bno', x_ft[:, :self.n_modes//2+1, :], weights)
        out_ft = out_ft.at[..., :self.n_modes//2+1, :].set(x_ft)

        out = jnp.fft.irfft(out_ft, axis=-2, n=out_grid_sz, norm=self.fft_norm)

        return out
    
    
class TimeModulatedSpectralConv1D(nn.Module):
    """ Time modulated integral kernel operator proposed by ``Learning PDE Solution Operator for Continuous Modelling of Time-Series`` """
    in_co_dim: int
    out_co_dim: int
    t_emb_dim: int
    n_modes: int
    out_grid_sz: int = None
    fft_norm: str = "forward"

    @nn.compact
    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray) -> jnp.ndarray:
        """ x shape: (batch, in_grid_sz, in_co_dim),
            t_emb shape: (batch, t_emb_dim)

            output shape: (batch, out_grid_sz, out_co_dim)
        """
        b, in_grid_sz, _ = x.shape
        out_grid_sz = in_grid_sz if self.out_grid_sz is None else self.out_grid_sz
        weights_shape = (self.n_modes//2+1, self.in_co_dim, self.out_co_dim)
        weights_real = self.param(
            'weights(real)',
            normal_initializer(self.in_co_dim),
            weights_shape,
        )
        weights_imag = self.param(
            'weights(imag)',
            normal_initializer(self.in_co_dim),
            weights_shape,
        )
        weights = weights_real + 1j*weights_imag

        x_ft = jnp.fft.rfft(x, axis=-2, norm=self.fft_norm)

        out_ft = jnp.zeros((b, in_grid_sz//2+1, self.out_co_dim), dtype=jnp.complex64)

        t_emb_transf_real = nn.Dense(
            self.n_modes//2+1,
            use_bias=False,
        )(t_emb)
        t_emb_transf_imag = nn.Dense(
            self.n_modes//2+1,
            use_bias=False,
        )(t_emb)
        t_emb_transf = t_emb_transf_real + 1j*t_emb_transf_imag

        weights = jnp.einsum('bn,nio->bnio', t_emb_transf, weights)
        x_ft = jnp.einsum('bni,bnio->bno', x_ft[:, :self.n_modes//2+1, :], weights)
        out_ft = out_ft.at[..., :self.n_modes//2+1, :].set(x_ft)

        out = jnp.fft.irfft(out_ft, axis=-2, n=out_grid_sz, norm=self.fft_norm)

        return out
    
class PointwiseOp1D(nn.Module):
    out_co_dim: int
    out_grid_sz: int = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(features=self.out_co_dim, kernel_size=(1,), padding='VALID')(x)
        if self.out_grid_sz is not None:
            in_grid = jnp.linspace(0, 1, x.shape[1])
            out_grid = jnp.linspace(0, 1, self.out_grid_sz)
            x = jax.vmap(
                jax.vmap(
                    jnp.interp,
                    in_axes=(None, None, -1),
                    out_axes=-1,
                    ),
                    in_axes=(None, None, 0),
                    out_axes=0,
                )(out_grid, in_grid, x)
        return x

### FNO Blocks ###
class TimeModulatedFourierBlock1D(nn.Module):
    in_co_dim: int
    out_co_dim: int
    t_emb_dim: int
    n_modes: int
    out_grid_sz: int = None
    fft_norm: str = "forward"
    norm: str = "batch"
    act: str = "relu"
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """ x shape: (batch, in_grid_sz, in_co_dim),
            t_emb shape: (batch, t_emb_dim)

            output shape: (batch, out_grid_sz, out_co_dim)
        """
        x_spec_out = TimeModulatedSpectralConv1D(
            self.in_co_dim,
            self.out_co_dim,
            self.t_emb_dim,
            self.n_modes,
            self.out_grid_sz,
            self.fft_norm
        )(x, t_emb)
        x_res_out = PointwiseOp1D(
            self.out_co_dim,
            self.out_grid_sz,
        )(x)
        x_out = x_spec_out + x_res_out
        if self.norm.lower() == "batch":
            x_out = nn.BatchNorm(use_running_average=not train)(x_out)
        elif self.norm.lower() == "instance":
            x_out = nn.InstanceNorm()(x_out)

        return get_activation_fn(self.act)(x_out)
    
    
class TimeEmbedding(nn.Module):
    """ Sinusoidal time step embedding """
    t_emb_dim: int
    scaling: float = 100.0
    max_period: float = 10000.0

    @nn.compact
    def __call__(self, t):
        """ t shape: (batch,) """
        pe = jnp.empty((len(t), self.t_emb_dim))
        factor = self.scaling * jnp.einsum('i,j->ij', 
                                           t, 
                                           jnp.exp(jnp.arange(0, self.t_emb_dim, 2) 
                                                   * -(jnp.log(self.max_period) / self.t_emb_dim)))
        pe = pe.at[:, 0::2].set(jnp.sin(factor))
        pe = pe.at[:, 1::2].set(jnp.cos(factor))
        return pe
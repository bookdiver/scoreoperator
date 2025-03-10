from functools import partial
from typing import Tuple, Optional
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
    """ Integral kernel operator for mapping functions (u: R -> R^{in_co_dim}) to functions (v: R -> R^{out_co_dim}) """
    in_co_dim: int
    out_co_dim: int
    n_modes: Tuple[int]
    out_grid_scaling: Optional[float] = 1.0
    fft_norm: str = "forward"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """ x shape: (batch, in_grid_sz, in_co_dim) 

            output shape: (batch, out_grid_sz, out_co_dim)
        """
        b, *in_grid_szs, _ = x.shape
        out_grid_szs = tuple(int(self.out_grid_scaling * in_grid_sz) for in_grid_sz in in_grid_szs)
        weights_shape = (self.n_modes[0]//2+1, self.in_co_dim, self.out_co_dim)
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

        out_ft = jnp.zeros((b, in_grid_szs[0]//2+1, self.out_co_dim), dtype=jnp.complex64)
        x_ft = jnp.einsum("bij,ijk->bik", x_ft[:, :self.n_modes[0]//2+1, :], weights)
        out_ft = out_ft.at[..., :self.n_modes[0]//2+1, :].set(x_ft)

        out = jnp.fft.irfft(out_ft, axis=-2, n=out_grid_szs[0], norm=self.fft_norm)
        
        return out
    
class SpectralConv2D(nn.Module):
    """ Integral kernel operator for mapping functions (u: R^2 -> R^{in_co_dim}) to functions (v: R^2 -> R^{out_co_dim}) """
    in_co_dim: int
    out_co_dim: int
    n_modes: Tuple[int, int]
    out_grid_scaling: Optional[float] = 1.0
    fft_norm: str = "forward"
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """ x shape: (batch, in_grid_sz[0], in_grid_sz[1], in_co_dim)
        
            output shape: (batch, out_grid_sz[0], out_grid_sz[1], out_co_dim)
        """
        b, *in_grid_szs, _ = x.shape
        out_grid_szs = tuple(int(self.out_grid_scaling * in_grid_sz) for in_grid_sz in in_grid_szs)
        weights_shape = (self.n_modes[0], self.n_modes[1]//2+1, self.in_co_dim, self.out_co_dim)
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
        
        x_ft = jnp.fft.rfft2(x, axes=(-2, -3), norm=self.fft_norm)
        
        out_ft = jnp.zeros((b, in_grid_szs[0], in_grid_szs[1]//2+1, self.out_co_dim), dtype=jnp.complex64)
        x_ft = jnp.einsum("bijk,ijkl->bijl", x_ft[:, :self.n_modes[0], :self.n_modes[1]//2+1, :], weights)
        out_ft = out_ft.at[..., :self.n_modes[0], :self.n_modes[1]//2+1, :].set(x_ft)
        
        out = jnp.fft.irfft2(out_ft, axes=(-2, -3), s=out_grid_szs, norm=self.fft_norm)
        
        return out
    
    
class FMSpectralConv1D(nn.Module):
    """ Frequency modulated integral kernel operator proposed by ``Learning PDE Solution Operator for Continuous Modelling of Time-Series`` """
    in_co_dim: int
    out_co_dim: int
    t_emb_dim: int
    n_modes: Tuple[int]
    out_grid_scaling: Optional[float] = 1.0
    fft_norm: str = "forward"

    @nn.compact
    def __call__(self, x: jnp.ndarray, phi_t: jnp.ndarray) -> jnp.ndarray:
        """ x shape: (batch, in_grid_sz, in_co_dim),
            phi_t shape: (batch, t_emb_dim)

            output shape: (batch, out_grid_sz, out_co_dim)
        """
        b, *in_grid_szs, _ = x.shape
        out_grid_szs = tuple(int(self.out_grid_scaling * in_grid_sz) for in_grid_sz in in_grid_szs)
        weights_shape = (self.n_modes[0]//2+1, self.in_co_dim, self.out_co_dim)
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

        out_ft = jnp.zeros((b, in_grid_szs[0]//2+1, self.out_co_dim), dtype=jnp.complex64)

        phi_t_ft_real = nn.Dense(
            self.n_modes[0]//2+1,
            use_bias=False,
        )(phi_t)
        phi_t_ft_imag = nn.Dense(
            self.n_modes[0]//2+1,
            use_bias=False,
        )(phi_t)
        phi_t_ft = phi_t_ft_real + 1j*phi_t_ft_imag

        weights = jnp.einsum("bij,jkl->bikl", phi_t_ft[:, :, None]*jnp.eye(self.n_modes[0]//2+1), weights)
        x_ft = jnp.einsum("bij,bijk->bik", x_ft[:, :self.n_modes[0]//2+1, :], weights)
        out_ft = out_ft.at[..., :self.n_modes[0]//2+1, :].set(x_ft)

        out = jnp.fft.irfft(out_ft, axis=-2, n=out_grid_szs[0], norm=self.fft_norm)
        return out
    
class FMSpectralConv2D(nn.Module):
    """ Frequency modulated integral kernel operator proposed by ``Learning PDE Solution Operator for Continuous Modelling of Time-Series`` """
    in_co_dim: int
    out_co_dim: int
    t_emb_dim: int
    n_modes: Tuple[int, int]
    out_grid_scaling: Optional[float] = 1.0
    fft_norm: str = "forward"
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, phi_t: jnp.ndarray) -> jnp.ndarray:
        """ x shape: (batch, in_grid_sz[0], in_grid_sz[1], in_co_dim),
            phi_t shape: (batch, t_emb_dim)

            output shape: (batch, out_grid_sz[0], out_grid_sz[1], out_co_dim)
        """
        b, *in_grid_szs, _ = x.shape
        out_grid_szs = tuple(int(self.out_grid_scaling * in_grid_sz) for in_grid_sz in in_grid_szs)
        weights_shape = (self.n_modes[0], self.n_modes[1]//2+1, self.in_co_dim, self.out_co_dim)
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
        
        x_ft = jnp.fft.rfft2(x, axes=(-3, -2), norm=self.fft_norm)
        
        out_ft = jnp.zeros((b, in_grid_szs[0], in_grid_szs[1]//2+1, self.out_co_dim), dtype=jnp.complex64)
        
        phi_t_ft_real = nn.Dense(
            int(self.n_modes[0]*(self.n_modes[1]//2+1)),
            use_bias=False, 
        )(phi_t)
        phi_t_ft_imag = nn.Dense(
            int(self.n_modes[0]*(self.n_modes[1]//2+1)),
            use_bias=False,
        )(phi_t)
        phi_t_ft = phi_t_ft_real + 1j*phi_t_ft_imag
        
        weights = weights.reshape(int(self.n_modes[0]*(self.n_modes[1]//2+1)), self.in_co_dim, self.out_co_dim)
        weights = jnp.einsum("bij,jkl->bikl", phi_t_ft[:, :, None]*jnp.eye(int(self.n_modes[0]*(self.n_modes[1]//2+1))), weights)
        weights = weights.reshape(b, self.n_modes[0], self.n_modes[1]//2+1, self.in_co_dim, self.out_co_dim)
        
        x_ft = jnp.einsum("bijk,bijkl->bijl", x_ft[:, :self.n_modes[0], :self.n_modes[1]//2+1, :], weights)
        out_ft = out_ft.at[..., :self.n_modes[0], :self.n_modes[1]//2+1, :].set(x_ft)
        
        out = jnp.fft.irfft2(out_ft, axes=(-3, -2), s=out_grid_szs, norm=self.fft_norm)
        
        return out
        
    
class TimeConv1D(nn.Module):
    out_co_dim: int
    out_grid_scaling: Optional[float] = 1.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, psi_t: jnp.ndarray) -> jnp.ndarray:
        """ x shape: (batch, in_grid_sz, in_co_dim),
            psi_tshape: (batch, t_emb_dim)
            
            output shape: (batch, out_grid_sz, out_co_dim)
        """
        _, *in_grid_szs, _ = x.shape
        out_grid_szs = tuple(int(self.out_grid_scaling * in_grid_sz) for in_grid_sz in in_grid_szs)
        x = nn.Conv(features=self.out_co_dim, kernel_size=(1,), padding="VALID")(x)
        weights = self.param(
            'weights',
            nn.initializers.normal(),
            (self.out_co_dim, self.out_co_dim)
        )
        psi_t = nn.Dense(
            2 * self.out_co_dim,
            use_bias=False
        )(psi_t)
        w_t, b_t = jnp.split(psi_t, 2, axis=-1) # (b, out_co_dim)
        x = jnp.einsum("ij,bjk,blk->bli", weights, w_t[:, :, None]*jnp.eye(self.out_co_dim), x)
        x = x + b_t[:, None, :]     # (b, in_grid_sz, out_co_dim)
        if self.out_grid_scaling != 1.0:
            x = jax.vmap(
                jax.vmap(
                    partial(jax.image.resize, shape=out_grid_szs, method="nearest"),
                    in_axes=-1,
                    out_axes=-1
                ),
                in_axes=0,
                out_axes=0
            )(x)        # (b, out_grid_sz, out_co_dim)
        return x
    
class TimeConv2D(nn.Module):
    out_co_dim: int
    out_grid_scaling: Optional[float] = 1.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, psi_t: jnp.ndarray) -> jnp.ndarray:
        """ x shape: (batch, in_grid_sz[0], in_grid_sz[1], in_co_dim),
            psi_t shape: (batch, t_emb_dim)
            
            output shape: (batch, out_grid_sz[0], out_grid_sz[1], out_co_dim)
        """
        b, *in_grid_szs, _ = x.shape
        out_grid_szs = tuple(int(self.out_grid_scaling * in_grid_sz) for in_grid_sz in in_grid_szs)
        x = nn.Conv(features=self.out_co_dim, kernel_size=(1, 1), padding="VALID")(x)
        weights = self.param(
            'weights',
            nn.initializers.normal(),
            (self.out_co_dim, self.out_co_dim)
        )
        psi_t = nn.Dense(
            2 * self.out_co_dim,
            use_bias=False
        )(psi_t)
        w_t, b_t = jnp.split(psi_t, 2, axis=-1) # (b, out_co_dim)
        x = x.reshape(b, -1, self.out_co_dim)
        x = jnp.einsum("ij,bjk,blk->bli", weights, w_t[:, :, None]*jnp.eye(self.out_co_dim), x)
        x = x + b_t[:, None, :]
        x = x.reshape(b, *in_grid_szs, self.out_co_dim)
        if self.out_grid_scaling != 1.0:
            x = jax.vmap(
                jax.vmap(
                    partial(jax.image.resize, shape=out_grid_szs, method="nearest"),
                    in_axes=-1,
                    out_axes=-1
                ),
                in_axes=0,
                out_axes=0
            )(x)
        return x

### FNO Blocks ###
class CTUNOBlock(nn.Module):
    do_dim: int
    in_co_dim: int
    out_co_dim: int
    t_emb_dim: int
    n_modes: Tuple[int, ...]
    out_grid_scaling: Optional[float] = 1.0
    fft_norm: str = "forward"
    norm: str = "batch"
    act: str = "relu"
    use_freq_mod: bool = True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """ x shape: (batch, in_grid_sz, in_co_dim),
            t_emb shape: (batch, t_emb_dim)

            output shape: (batch, out_grid_sz, out_co_dim)
        """
        if self.use_freq_mod:
            if self.do_dim == 1:
                spec_conv_klass = FMSpectralConv1D
                time_conv_klass = TimeConv1D
            elif self.do_dim == 2:
                spec_conv_klass = FMSpectralConv2D
                time_conv_klass = TimeConv2D
            else:
                raise ValueError(f"Invalid do_dim: {self.do_dim}")
            
            x_spec_out = spec_conv_klass(
                self.in_co_dim,
                self.out_co_dim,
                self.t_emb_dim,
                self.n_modes,
                self.out_grid_scaling,
                self.fft_norm
            )(x, t_emb)
            x_res_out = time_conv_klass(
                self.out_co_dim,
                self.out_grid_scaling,
            )(x, t_emb)
            
        else:
            if self.do_dim == 1:
                spec_conv_klass = SpectralConv1D
                time_conv_klass = TimeConv1D
            elif self.do_dim == 2:
                spec_conv_klass = SpectralConv2D
                time_conv_klass = TimeConv2D
            else:
                raise ValueError(f"Invalid do_dim: {self.do_dim}")
            x_spec_out = spec_conv_klass(
                self.in_co_dim,
                self.out_co_dim,
                self.n_modes,
                self.out_grid_scaling,
                self.fft_norm
            )(x)
            x_res_out = time_conv_klass(
                self.out_co_dim,
                self.out_grid_scaling,
            )(x, t_emb)
        
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
    max_period: float = 128.0

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
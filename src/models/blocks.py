import jax.numpy as jnp
from flax import linen as nn

class SpectralConv1D(nn.Module):
    """ Integral kernel operator for mapping functions (u: R -> R^{input_dim}) to functions (v: R -> R^{output_dim}) """
    input_dim: int
    output_dim: int
    n_modes: int

    def setup(self):
        weights_R_shape = (self.n_modes//2+1, self.input_dim, self.output_dim)
        self.weights_R_real = self.param(
            'weights_R(real)',
            nn.initializers.xavier_normal(),
            weights_R_shape
        )
        self.weights_R_imag = self.param(
            'weights_R(imag)',
            nn.initializers.xavier_normal(),
            weights_R_shape
        )
    
    def __call__(self, x):
        """ x shape: (batch, n_samples, input_dim) """
        x_ft = jnp.fft.rfft(x, axis=-2)
        out_ft = jnp.zeros((*x.shape[:-2], x.shape[-2]//2+1, self.output_dim), dtype=jnp.complex64)
        x_ft_transf = jnp.einsum('...ni,nio->...no', x_ft[..., :self.n_modes//2+1, :], self.weights_R_real+1j*self.weights_R_imag)
        out_ft = out_ft.at[..., :self.n_modes//2+1, :].set(x_ft_transf)
        out = jnp.fft.irfft(out_ft, axis=-2)
        return out
    
class TMSpectralConv1D(nn.Module):
    """ Time moduled integral kernel operator proposed by ``Learning PDE Solution Operator for Continuous Modelling of Time-Series`` """
    input_dim: int
    output_dim: int
    encoding_dim: int
    n_modes: int

    def setup(self):
        weights_R_shape = (self.n_modes//2+1, self.input_dim, self.output_dim)
        self.weights_R_real = self.param(
            'weights_R(real)',
            nn.initializers.xavier_normal(),
            weights_R_shape,
        )
        self.weights_R_imag = self.param(
            'weights_R(imag)',
            nn.initializers.xavier_normal(),
            weights_R_shape,
        )
        self.t_dense_real = nn.Dense(self.n_modes//2+1, use_bias=False)
        self.t_dense_imag = nn.Dense(self.n_modes//2+1, use_bias=False)

    def __call__(self, x, t_emb):
        """ x shape: (batch, n_samples, input_dim),
            t_emb shape: (batch, encoding_dim)
        """
        x_ft = jnp.fft.rfft(x, axis=-2)
        out_ft = jnp.zeros((*x.shape[:-2], x.shape[-2]//2+1, self.output_dim), dtype=jnp.complex64)
        t_emb_transf = self.t_dense_real(t_emb) + 1j*self.t_dense_imag(t_emb)
        weights = jnp.einsum('...n,nio->...nio', t_emb_transf, self.weights_R_real+1j*self.weights_R_imag)
        x_ft_transf = jnp.einsum('...ni,...nio->...no', x_ft[..., :self.n_modes//2+1, :], weights)
        out_ft = out_ft.at[..., :self.n_modes//2+1, :].set(x_ft_transf)
        out = jnp.fft.irfft(out_ft, axis=-2)
        return out
    
class FourierBlock1D(nn.Module):
    """ Original Fourier block in  ``Fourier Neural Operator for Parametric Partial Differential Equations`` """
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
        """ x shape: (batch, n_samples, input_dim) """ 
        x_res = self.spectral_conv(x)
        x_jump = self.residual_transf(x)
        return self.activation(x_res + x_jump)
    
class ResidualFourierBlock1D(nn.Module):
    """ Modified Fourier block to incorporate time step embedding, used by ``Score-based Diffusion Models in Function Space paper`` """
    input_dim: int
    output_dim: int
    encoding_dim: int
    n_modes: int
    activation: nn.activation = nn.relu

    def setup(self):
        self.spectral_conv1 = SpectralConv1D(
            self.input_dim,
            self.output_dim,
            self.n_modes,
        )
        self.spectral_conv2 = SpectralConv1D(
            self.output_dim,
            self.output_dim,
            self.n_modes,
        )
        self.residual_transf = nn.Dense(
            self.output_dim,
            kernel_init=nn.initializers.xavier_normal(),
        )
        self.time_mlp = nn.Dense(
            2 * self.output_dim,
            kernel_init=nn.initializers.xavier_normal(),
        )
    
    def __call__(self, x, t_emb):
        """ x shape: (batch, n_samples, input_dim),
            t_emb shape: (batch, encoding_dim)
        """
        x_res = self.activation(self.spectral_conv1(x))
        w, b = jnp.split(self.time_mlp(t_emb)[:, None, :], 2, axis=-1)
        x_res = x_res * (w + 1.0) + b
        x_res = self.spectral_conv2(self.activation(x_res))
        x_jump = self.residual_transf(x)
        return x_res + x_jump
    
class TMFourierBlock1D(nn.Module):
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
        """ x shape: (batch, n_samples, input_dim),
            t_emb shape: (batch, encoding_dim)
        """
        t_emb = self.t_dense(t_emb)
        x_res = jnp.einsum('...d,...nd->...nd', t_emb, x)
        return self.activation(self.spectral_conv(x, t_emb) + self.residual_transf(x_res))
    
class TimeEncoding(nn.Module):
    """ Sinusoidal time step embedding """
    encoding_dim: int
    scaling: float = 100.0
    
    def setup(self):
        self.div_term = jnp.exp(jnp.arange(0, self.encoding_dim, 2) * -(jnp.log(10000.0) / self.encoding_dim))

    def __call__(self, t):
        """ t shape: (batch,) """
        pe = jnp.empty((len(t), self.encoding_dim))
        factor = self.scaling * jnp.einsum('i,j->ij', t, self.div_term)
        pe = pe.at[:, 0::2].set(jnp.sin(factor))
        pe = pe.at[:, 1::2].set(jnp.cos(factor))
        return pe
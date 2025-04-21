import jax
import jax.numpy as jnp
from flax import linen as nn

##############################
#### Spetral convolutions ####
##############################
class SpectralFMConv1D(nn.Module):
    """ 
    Time modulated integral kernel operator proposed by https://arxiv.org/abs/2302.00854,
    used as an operator maps f: R -> R^{d_u} to g: R -> R^{d_v}
    """
    modes: int          # number of low-frequency modes
    d_u: int            # input channel dimension
    d_v: int            # output channel dimension
    c: int              # time embedding dimension

    @nn.compact
    def __call__(self, x, phi_t):
        """ 
        x:          (b, L, d_u),
        phi_t:      (b, c)

        Returns:    (b, L, d_v)
        """
        b, L, _ = x.shape
        
        # 1) forward real-to-complex FFT (real)
        x_ft = jnp.fft.rfft(x, axis=1)      # (b, L//2+1, d_u)
        x_ft = x_ft[:, :self.modes, :]      # (b, modes, d_u)
        
        # 2) build time-dependent factors phi_l(t)(freq) for freq in [0, ..., modes-1]
        A_real = self.param('A_real', nn.initializers.normal(0.01), (self.modes, self.c))
        A_imag = self.param('A_imag', nn.initializers.normal(0.01), (self.modes, self.c))
        
        # broadcast A_real (modes, c), A_imag (modes, c) and phi_t (b, c) to allow batch multiplication, 
        # we want to obtain output A * phi_t (b, modes)
        A_real_exp = A_real[None, :, :]     # (1, modes, c)
        A_imag_exp = A_imag[None, :, :]     # (1, modes, c)
        phi_t_exp = phi_t[:, None, :]       # (b, 1, c)
        
        # do the complex multiplication individually
        phi_real = jnp.einsum('bmc, bmc -> bm', A_real_exp, phi_t_exp)
        phi_imag = jnp.einsum('bmc, bmc -> bm', A_imag_exp, phi_t_exp)
        phi_complex = phi_real + 1j * phi_imag  # (b, modes)
        
        # 3) base fourier weights, stored as real & imaginary parts individually
        R_real = self.param('R_real', nn.initializers.normal(0.01), (self.modes, self.d_v, self.d_u))
        R_imag = self.param('R_imag', nn.initializers.normal(0.01), (self.modes, self.d_v, self.d_u))
        
        def freq_multiply(args):
            """ This mulitplication shall be vectorized over b and modes dimensions
            """
            x_slice, phi_val, Rr, Ri = args # x_slice (d_u, ), phi_val (,), Rr & Ri (d_v, d_u)
            xr, xi = x_slice.real, x_slice.imag
            # first multiply on the fourier base R F(v)
            # complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i 
            real_tmp = Rr @ xr - Ri @ xi    # (d_v, )
            imag_tmp = Rr @ xi + Ri @ xr    # (d_v, )
            
            # then multiply by the time information
            phi_real, phi_imag = phi_val.real, phi_val.imag
            real_out = real_tmp * phi_real - imag_tmp * phi_imag    # (d_v, )
            imag_out = real_tmp * phi_imag + imag_tmp * phi_real    # (d_v, )
            out = real_out + 1j * imag_out
            return out
        
        # we need to vectorize freq_multiply() over b and modes, a simple way is to flatten them and vectorize once
        bm = b * self.modes
        x_ft_flat = x_ft.reshape((bm, self.d_u))
        phi_flat = phi_complex.reshape((bm, ))
        Rr_flat = jnp.repeat(R_real, b, axis=0)     # (b*modes, d_v, d_u)
        Ri_flat = jnp.repeat(R_imag, b, axis=0)     # (b*modes, d_v, d_u)
        freq_args = (x_ft_flat, phi_flat, Rr_flat, Ri_flat)
        
        out_flat = jax.vmap(freq_multiply)(freq_args)   # (b*modes, d_v)
        out_ft = out_flat.reshape((b, self.modes, self.d_v))
        
        # 4) zero-padding to L//2+1 for irfft
        full_ft = jnp.zeros((b, L//2 + 1, self.d_v), dtype=jnp.complex64)
        full_ft = full_ft.at[:, :self.modes, :].set(out_ft)
        x_out = jnp.fft.irfft(full_ft, n=L, axis=1)     # (b, L, d_v)
        return x_out
    
class SpectralFMConv2D(nn.Module):
    """ 
    Time modulated integral kernel operator proposed by https://arxiv.org/abs/2302.00854,
    used as an operator maps f: R^2 -> R^{d_u} to g: R^2 -> R^{d_v}
    """
    modes: int
    d_u: int
    d_v: int
    c: int

    @nn.compact
    def __call__(self, x, phi_t):
        """
        x:          (b, H, W, d_u)
        phi_t:      (b, c)

        Returns:    (b, H, W, d_v)
        """
        b, H, W, _ = x.shape
        
        # 1) forward real-to-complex FFT (real)
        x_ft = jnp.fft.rfft2(x, axes=(2, 1))    # (b, H, W//2+1, d_u)
        x_ft = x_ft[:, :self.modes, :self.modes, :]  # (b, modes, modes, d_u)
        
        # 2) build time-dependent factors phi_l(t)(freq) for freq in [0, ..., modes-1]
        A_real = self.param('A_real', nn.initializers.normal(0.01), (self.modes, self.modes, self.c))
        A_imag = self.param('A_imag', nn.initializers.normal(0.01), (self.modes, self.modes, self.c))
        
        # broadcast A_real (modes, modes, c), A_imag (modes, modes, c) and phi_t (b, c) to allow batch multiplication,
        # we want to obtain output A * phi_t (b, modes, modes)
        A_real_exp = A_real[None, :, :, :]     # (1, modes, modes, c)
        A_imag_exp = A_imag[None, :, :, :]     # (1, modes_h, modes_w, c)
        phi_t_exp = phi_t[:, None, None, :]    # (b, 1, 1, c)
        
        # do the complex multiplication individually
        phi_real = jnp.einsum('bmwc, bmwc -> bmw', A_real_exp, phi_t_exp)
        phi_imag = jnp.einsum('bmwc, bmwc -> bmw', A_imag_exp, phi_t_exp)
        phi_complex = phi_real + 1j * phi_imag  # (b, modes, modes)
        
        # 3) base fourier weights, stored as real & imaginary parts individually
        R_real = self.param('R_real', nn.initializers.normal(0.01), (self.modes, self.modes, self.d_v, self.d_u))
        R_imag = self.param('R_imag', nn.initializers.normal(0.01), (self.modes, self.modes, self.d_v, self.d_u))
        
        def freq_multiply(args):
            """ This multiplication shall be vectorized over b and modes dimensions
            """
            x_slice, phi_val, Rr, Ri = args # x_slice (d_u, ), phi_val (,), Rr & Ri (d_v, d_u)
            xr, xi = x_slice.real, x_slice.imag
            # first multiply on the fourier base R F(v)
            # complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
            real_tmp = Rr @ xr - Ri @ xi    # (d_v, )
            imag_tmp = Rr @ xi + Ri @ xr    # (d_v, )
            
            # then multiply by the time information
            phi_real, phi_imag = phi_val.real, phi_val.imag
            real_out = real_tmp * phi_real - imag_tmp * phi_imag    # (d_v, )
            imag_out = real_tmp * phi_imag + imag_tmp * phi_real    # (d_v, )
            out = real_out + 1j * imag_out
            return out
        
        # we need to vectorize freq_multiply() over b and modes, a simple way is to flatten them and vectorize once
        bhw = b * self.modes * self.modes
        x_ft_flat = x_ft.reshape((bhw, self.d_u))
        phi_flat = phi_complex.reshape((bhw, ))
        Rr_flat = jnp.repeat(R_real.reshape((self.modes * self.modes, self.d_v, self.d_u)), b, axis=0)     # (b*modes*modes, d_v, d_u)
        Ri_flat = jnp.repeat(R_imag.reshape((self.modes * self.modes, self.d_v, self.d_u)), b, axis=0)     # (b*modes*modes, d_v, d_u)
        freq_args = (x_ft_flat, phi_flat, Rr_flat, Ri_flat)
        
        out_flat = jax.vmap(freq_multiply)(freq_args)   # (b*modes*modes, d_v)
        out_ft = out_flat.reshape((b, self.modes, self.modes, self.d_v))
        
        # 4) zero-padding to H1, W//2+1 for irfft2
        full_ft = jnp.zeros((b, H, W//2 + 1, self.d_v), dtype=jnp.complex64)
        full_ft = full_ft.at[:, :self.modes, :self.modes, :].set(out_ft)
        x_out = jnp.fft.irfft2(full_ft, s=(H, W), axes=(2, 1))     # (b, H, W, d_v)
        return x_out

#######################
### Residual blocks ###
#######################
    
class Res1D(nn.Module):
    d_u: int        # input channel dimension
    d_v: int        # output channel dimension
    c: int          # time embedding dimension
    
    @nn.compact
    def __call__(self, x, psi_t):
        """
        x:          (b, L, d_u)
        psi_t:      (b, c)
        
        Returns:    (b, L, d_v)
        """
        # 1) build W_l(t) = W_l diag(B_l psi_t)
        W = self.param('W', nn.initializers.normal(0.01), (self.d_u, self.d_v))
        B = self.param('B', nn.initializers.normal(0.01), (self.d_v, self.c))
        
        # psi_t: (b, c)
        # B @ psi_t[b] => (d_v, ) => diag => (d_v, d_v)
        # => multiply W => Wt: (d_u, d_v)
        
        def build_Wt(psi_t_b):
            """ This function shall be vectorized over b dimension
            """
            diag_vec = B @ psi_t_b          # (d_v, )
            diag_mat = jnp.diag(diag_vec)   # (d_v, d_v)
            return W @ diag_mat             # (d_u, d_v)
        
        Wt = jax.vmap(build_Wt)(psi_t)      # (b, d_u, d_v)
        
        # 2) apply W_t = W_l(t) on x
        # x: (b, L, d_u), for each b, we do (L, d_u) x (d_u, d_v) => (L, d_v)
        def apply_Wt(args):
            x_b, wt_b = args    # x_b: (L, d_u), wt_b: (d_u, d_v)
            return x_b @ wt_b
        
        out = jax.vmap(apply_Wt)((x, Wt))
        return out
    
class Res2D(nn.Module):
    d_u: int
    d_v: int
    c: int
    
    @nn.compact
    def __call__(self, x, psi_t):
        """
        x:          (b, H, W, d_u)
        psi_t:      (b, c)

        Returns:    (b, H, W, d_v)
        """
        # 1) build W_l(t) = W_l diag(B_l psi_t)
        W = self.param('W', nn.initializers.normal(0.01), (self.d_u, self.d_v))
        B = self.param('B', nn.initializers.normal(0.01), (self.d_v, self.c))
        
        # psi_t: (b, c)
        # B @ psi_t[b] => (d_v, ) => diag => (d_v, d_v)
        # => muliply W => Wt: (d_u, d_v)
        
        def build_Wt(psi_t_b):
            """ This function shall be vectorized over b dimension
            """
            diag_vec = B @ psi_t_b          # (d_v, )
            diag_mat = jnp.diag(diag_vec)   # (d_v, d_v)
            return W @ diag_mat             # (d_u, d_v)
        
        Wt = jax.vmap(build_Wt)(psi_t)      # (b, d_u, d_v)
        
        # 2) apply W_t = W_l(t) on x
        # x: (b, H, W, d_u), for each b, we do (H, W, d_u) x (d_u, d_v) => (H, W, d_v)
        def apply_Wt(args):
            x_b, wt_b = args    # x_b: (H, W, d_u), wt_b: (d_u, d_v)
            return jnp.einsum('hwu, uv -> hwv', x_b, wt_b)   # (H, W, d_v)
        
        out = jax.vmap(apply_Wt)((x, Wt))
        return out

###################
### FNO Blocks ####
###################
class Block1D(nn.Module):
    modes: int
    d_u: int
    d_v: int
    c: int
    
    @nn.compact
    def __call__(self, x, phi_t, psi_t, train = True):
        """ 
        x:          (b, L, d_u),
        phi_t:      (b, c)
        psi_t:      (b, c)

        Returns:    (b, L, d_v)
        """
        x_spec_out = SpectralFMConv1D(
            self.modes, self.d_u, self.d_v, self.c
        )(x, phi_t)
        x_res_out = Res1D(
            self.d_u, self.d_v, self.c
        )(x, psi_t)
        x_out = x_spec_out + x_res_out
        
        # x_out = nn.BatchNorm(use_running_average=not train)(x_out)
        # x_out = nn.InstanceNorm()(x_out)
        
        x_out = nn.gelu(x_out)

        return x_out
    
class Block2D(nn.Module):
    modes: int
    d_u: int
    d_v: int
    c: int
    
    @nn.compact
    def __call__(self, x, phi_t, psi_t, train = True):
        """
        x:          (b, H, W, d_u)
        phi_t:      (b, c)
        psi_t:      (b, c)

        Returns:    (b, H, W, d_v)
        """
        x_spec_out = SpectralFMConv2D(
            self.modes, self.d_u, self.d_v, self.c
        )(x, phi_t)
        x_res_out = Res2D(
            self.d_u, self.d_v, self.c
        )(x, psi_t)
        x_out = x_spec_out + x_res_out
        
        # x_out = nn.BatchNorm(use_running_average=not train)(x_out)
        # x_out = nn.InstanceNorm()(x_out)

        x_out = nn.gelu(x_out)
        return x_out


########################################
#### Sinusoidal time step embedding ####
########################################
class TimeEmbedding(nn.Module):
    """ Sinusoidal time step embedding """
    c: int                          # time embedding dimension
    s: float = 100.0                # time scaling
    min_freq: float = 1.0           # minimal frequency
    max_freq: float = 10000.0       # maximal frequency

    @nn.compact
    def __call__(self, t):
        """ 
        t:          (b,)
        
        Returns:    (b, c) 
        """
        t = self.s * t[:, jnp.newaxis]
        num_freqs = self.c // 2
        freqs = 2.0 * jnp.pi * jnp.exp(
            jnp.linspace(
                jnp.log(self.min_freq),
                jnp.log(self.max_freq),
                num_freqs
            )
        )
        freqs = jax.lax.stop_gradient(freqs)
        arg = t * freqs
        sin_emb = jnp.sin(arg)
        cos_emb = jnp.cos(arg)
        embedding = jnp.stack([sin_emb, cos_emb], axis=-1)
        embedding = embedding.reshape(-1, self.c)
        return embedding
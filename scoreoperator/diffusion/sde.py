"""
Stochastic Differential Equation (SDE) implementations for diffusion bridges.

This module provides implementations of various SDEs used in diffusion bridges,
including base abstract class, Cylindrical Brownian motion SDE, and Kunita SDE.
"""
import abc
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from scoreoperator.utils.data import BaseData, Zero

class BaseSDE(abc.ABC):
    T: float
    x0: BaseData
    xT: BaseData

    def __init__(self, config):
        super().__init__()
        self.T = config['T']
        self.x0 = config['x0']
        self.xT = config['xT']
        
    @abc.abstractmethod
    def f(self, t, x):
        pass

    @abc.abstractmethod
    def g(self, t, x):
        pass
    
    @abc.abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def apply_g(self, Phi, dW):
        pass
    
    def project(self, x_shape, t_shape, w_shape):
        return ProjectedSDE(self.f, self.g, self.apply_g, self.x0, self.xT, self.T, t_shape, x_shape, w_shape)

    @staticmethod
    def get_reverse_projected_bridge(forward_sde, score, x_shape, t_shape, w_shape, xT = None):
        T = forward_sde.T
        x0 = forward_sde.x0
        xT = forward_sde.xT if xT is None else xT
        
        projected_sde = forward_sde.project(x_shape, t_shape, w_shape)
        
        reversed_f = lambda t, x: -projected_sde.f(T - t, x) + score(T - t, x)
        reversed_g = lambda t, x: projected_sde.g(T - t, x)
        reversed_apply_g = lambda Phi, dW: projected_sde.apply_g(Phi, dW)
        
        return ProjectedSDE(reversed_f, reversed_g, reversed_apply_g, xT, x0, T, t_shape, x_shape, w_shape)
        
class ProjectedSDE:
    f: callable
    g: callable
    apply_g: callable
    x0_: jnp.ndarray
    xT_: jnp.ndarray
    T: float
    t_shape: tuple
    x_shape: tuple
    w_shape: tuple
    ts: jnp.ndarray
    dt: jnp.ndarray
    
    def __init__(self, f, g, apply_g, x0, xT, T, t_shape, x_shape, w_shape):
        self.f = f
        self.g = g
        self.apply_g = apply_g
        self.T = T
        self.t_shape = t_shape
        self.x_shape = x_shape
        self.w_shape = w_shape
        self.x0_ = x0.eval(x_shape)
        self.xT_ = xT.eval(x_shape)
        
    @property
    def ts(self):
        return jnp.linspace(0., self.T, self.t_shape[0], endpoint=False)
    
    @property
    def dt(self):
        return self.T / self.t_shape[0]
    
class CylindricalBrownianSDE(BaseSDE):
    sigma: float
    
    def __init__(self, config):
        super().__init__(config)
        self.sigma = config["sigma"]

    def f(self, t, x):
        return jnp.zeros_like(x)
    
    def g(self, t, x):
        return None
    
    def apply_g(self, Phi, dW):
        return self.sigma * dW


class KunitaFlowSDE(BaseSDE):
    k_alpha: float
    k_sigma: float
    
    def __init__(self, config):
        super().__init__(config)
        self.k_alpha = config["k_alpha"]
        self.k_sigma = config["k_sigma"]
    
    def f(self, t, x):
        return jnp.zeros_like(x)
    
    def g(self, t, x):
        return x

    def apply_g(self, Phi, dW):
        return None
    
    def project(self, x_shape, t_shape, w_shape):
        w_span = (-2.0, 2.0)
        scaling = jnp.abs(w_shape[0] / (w_span[1] - w_span[0]))
        grid_spacing = 1.0 / scaling
        center = jnp.array([w_shape[0] / 2, w_shape[1] / 2])
        
        coords_to_pixels = lambda x: scaling * x + center[jnp.newaxis, :]
        
        def make_gaussian_kernel_fft(w_shape, k_sigma, grid_spacing):
            m, n = w_shape[:-1]
            x = jnp.fft.fftfreq(m, grid_spacing)
            y = jnp.fft.fftfreq(n, grid_spacing)
            xx, yy = jnp.meshgrid(x, y, indexing='ij')
            kernel_ft = jnp.exp(-2. * jnp.pi**2 * k_sigma**2 * (xx**2 + yy**2))
            kernel_ft = kernel_ft / jnp.sqrt(jnp.sum(jnp.abs(kernel_ft)**2) / kernel_ft.size)
            return kernel_ft
        
        kernel_ft = make_gaussian_kernel_fft(w_shape, self.k_sigma, grid_spacing)
        
        def g(t, x):
            return self.g(t, x) + self.x0.eval(x_shape)
        
        def apply_g(Phi, dW):
            dW_convolved = jnp.stack([
                jnp.fft.ifft2(kernel_ft * jnp.fft.fft2(dW[..., 0])).real,
                jnp.fft.ifft2(kernel_ft * jnp.fft.fft2(dW[..., 1])).real,
            ], axis=-1)
            dW_convolved = self.k_alpha * dW_convolved
            
            x_pixels = coords_to_pixels(Phi)  # shape (N, 2)
            x_coords = x_pixels.T  # shape (2, N)

            def interp_channel(channel):  # channel: shape (H, W)
                return jsp.ndimage.map_coordinates(channel, x_coords, order=1, mode="nearest")

            Phi_dW = jax.vmap(interp_channel, in_axes=2, out_axes=1)(dW_convolved)  # shape (C, N)
            
            return Phi_dW
        
        return ProjectedSDE(
            self.f, 
            g, 
            apply_g, 
            Zero(), 
            self.xT - self.x0, 
            self.T, 
            t_shape, 
            x_shape, 
            w_shape
        )

class SDEFactory:
    """
    Factory class for creating different SDE instances.
    
    Provides a centralized way to create SDE objects based on type specification.
    """
    @staticmethod
    def create(sde_type, sde_config):
        """
        Create an SDE instance of the specified type.
        
        Args:
            sde_type (str): Type of SDE to create ('cylindrical_brownian', 'kunita_flow').
            sde_config (dict): Configuration for the SDE.
            
        Returns:
            BaseSDE: The created SDE instance.
            
        Raises:
            ValueError: If sde_type is not recognized.
        """
        sde_classes = {
            "zero": Zero,
            "cylindrical_brownian": CylindricalBrownianSDE,
            "kunita_flow": KunitaFlowSDE,
        }

        if sde_type.lower() not in sde_classes:
            raise ValueError(f"Unknown SDE type: {sde_type}")

        sde_class = sde_classes[sde_type.lower()]
        return sde_class(sde_config)
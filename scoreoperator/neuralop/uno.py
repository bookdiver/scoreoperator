import jax.numpy as jnp
from flax import linen as nn

from scoreoperator.neuralop.blocks import (
    Block1D, Block2D, TimeEmbedding
)

class CTUNO1D(nn.Module):
    """ 
    Continuous-time U-shaped Fourier neural operator that maps
    [0, T] x L^2(R, R^{d_u}) to L^2(R, R^{d_v}).
    """
    d_u: int                        # input channel dimensions
    d_v: int                        # output channel dimensions
    c: int                          # time embedding dimensions
    d_ls: tuple[int]                # intermediate channel dimensions 
    modes_ls: tuple[int]            # intermediate number of low frequencies

    @nn.compact
    def __call__(self, x, t, train = False):
        """
        x:          (b, L, d_u)
        t:          (b, )
        
        Returns:    (b, L, d_v)
        """
        assert len(self.d_ls) == len(self.modes_ls) + 1
        phi_t = TimeEmbedding(self.c)(t)
        psi_t = TimeEmbedding(self.c)(t)
        
        downs = []
        
        # lifting
        d_lifting = self.d_ls[0]
        x = nn.Dense(d_lifting, 
                     kernel_init=nn.initializers.normal(0.01),
                    )(x)
        
        # down path
        for i in range(len(self.modes_ls)):
            x = Block1D(
                modes=self.modes_ls[i],
                d_u=self.d_ls[i],
                d_v=self.d_ls[i+1],
                c=self.c,
                name=f"DownLayer{i}"
            )(x, phi_t, psi_t, train)
            downs.append(x)
        
        # bottleneck
        x = Block1D(
            modes=self.modes_ls[-1],
            d_u=self.d_ls[-1],
            d_v=self.d_ls[-1],
            c=self.c,
            name="BottleNeckLayer"
        )(x, phi_t, psi_t, train)
        
        # up path
        for i in reversed(range(len(self.modes_ls))):
            x_down = downs.pop()
            x = jnp.concatenate([x, x_down], axis=-1)
            x = Block1D(
                modes=self.modes_ls[i],
                d_u=self.d_ls[i+1] * 2,
                d_v=self.d_ls[i],
                c=self.c,
                name=f"UpLayer{i}"
            )(x, phi_t, psi_t, train)
            
        # projection
        x = nn.Dense(self.d_v, 
                     kernel_init=nn.initializers.normal(0.01),
                    )(x)
        
        return x

class CTUNO2D(nn.Module):
    """ 
    Continuous-time U-shaped Fourier neural operator that maps
    [0, T] x L^2(R^2, R^{d_u}) to L^2(R^2, R^{d_v}).
    """
    d_u: int                        # input channel dimensions
    d_v: int                        # output channel dimensions
    c: int                          # time embedding dimensions
    d_ls: tuple[int]                # intermediate channel dimensions 
    modes_ls: tuple[int]            # intermediate number of low frequencies

    @nn.compact
    def __call__(self, x, t, train = False):
        """
        x:          (b, H, W, d_u)
        t:          (b, )
        
        Returns:    (b, H, W, d_v)
        """
        assert len(self.d_ls) == len(self.modes_ls) + 1
        phi_t = TimeEmbedding(self.c)(t)
        psi_t = TimeEmbedding(self.c)(t)
        
        downs = []
        
        # lifting
        d_lifting = self.d_ls[0]
        x = nn.Dense(d_lifting, 
                     kernel_init=nn.initializers.normal(0.01),
                    )(x)
        
        # down path
        for i in range(len(self.modes_ls)):
            x = Block2D(
                modes=self.modes_ls[i],
                d_u=self.d_ls[i],
                d_v=self.d_ls[i+1],
                c=self.c,
                name=f"DownLayer{i}"
            )(x, phi_t, psi_t, train)
            downs.append(x)
        
        # bottleneck
        x = Block2D(
            modes=self.modes_ls[-1],
            d_u=self.d_ls[-1],
            d_v=self.d_ls[-1],
            c=self.c,
            name="BottleNeckLayer"
        )(x, phi_t, psi_t, train)
        
        # up path
        for i in reversed(range(len(self.modes_ls))):
            x_down = downs.pop()
            x = jnp.concatenate([x, x_down], axis=-1)
            x = Block2D(
                modes=self.modes_ls[i],
                d_u=self.d_ls[i+1] * 2,
                d_v=self.d_ls[i],
                c=self.c,
                name=f"UpLayer{i}"
            )(x, phi_t, psi_t, train)
            
        # projection
        x = nn.Dense(self.d_v, 
                     kernel_init=nn.initializers.normal(0.01),
                    )(x)
        return x
        
class ModelFactory:
    
    @staticmethod
    def create(model_type, model_config):
        model_classes = {
            "CTUNO1D": CTUNO1D,
            "CTUNO2D": CTUNO2D
        }
        
        if model_type not in model_classes:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = model_classes[model_type]
        return model_class(**model_config)



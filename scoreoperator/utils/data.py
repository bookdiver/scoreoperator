import abc
import os
import numpy as np
from scipy.interpolate import interp1d
import jax.numpy as jnp

class BaseData(abc.ABC):

    @abc.abstractmethod
    def eval(self, shape):
        pass
        
class ToyFunction(BaseData):
    """ Toy quadratic function: f(x) = ax^2 + c, a function R->R"""
    a: float
    c: float

    def __init__(self, a, c):
        self.a = a
        self.c = c
    
    def eval(self, shape):
        evaluation = jnp.expand_dims(jnp.linspace(-1.0, 1.0, shape[0]) ** 2 * self.a + self.c, axis=-1)
        assert evaluation.shape == shape
        return evaluation
    

class Ellipse(BaseData):
    """ Ellipse shape data, which is treated as a function R->R^2, i.e. parametric equations"""
    a: float
    b: float
    shift_x: float
    shift_y: float

    def __init__(self, a, b, shift_x, shift_y):
        self.a = a
        self.b = b
        self.shift_x = shift_x
        self.shift_y = shift_y
    
    def eval(self, shape):
        t = jnp.linspace(0, 2*jnp.pi, shape[0], endpoint=False)
        x = self.a * jnp.cos(t) + self.shift_x
        y = self.b * jnp.sin(t) + self.shift_y
        evaluation = jnp.stack([x, y], axis=1)
        assert evaluation.shape == shape
        return evaluation

    
class Ellipsoid(BaseData):
    """ Ellipsoid shape data, which is treated as a function R->R^3"""
    a: float
    b: float
    c: float
    shift_x: float
    shift_y: float
    shift_z: float
    
    def __init__(self, a, b, c, shift_x, shift_y, shift_z):
        self.a = a
        self.b = b
        self.c = c
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.shift_z = shift_z
    
    def eval(self, shape) -> jnp.ndarray:
        # Calculate the number of points for each dimension
        
        # Generate spherical coordinates
        theta = jnp.linspace(0, jnp.pi, shape[0])
        phi = jnp.linspace(0, 2*jnp.pi, shape[1])
        
        # Create a meshgrid of theta and phi
        theta, phi = jnp.meshgrid(theta, phi)
        
        # Convert spherical coordinates to Cartesian coordinates
        x = self.a * jnp.sin(theta) * jnp.cos(phi) + self.shift_x
        y = self.b * jnp.sin(theta) * jnp.sin(phi) + self.shift_y
        z = self.c * jnp.cos(theta) + self.shift_z
        
        # Reshape the coordinates into a 2D array of shape (n_pts[0], n_pts[1], 3)
        evaluation = jnp.stack([x, y, z], axis=-1)
        assert evaluation.shape == shape
        return evaluation

class Butterfly(BaseData):
    raw_pts: jnp.ndarray

    def __init__(self, name):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, "../../landmarks_raw/normalized", f"{name}.npy")
        path = os.path.normpath(path)
        
        try:
            self.raw_pts = np.load(path)
        except Exception as e:
            print(f"Error loading {name}: {e}")
            
    def eval(self, shape):
        do_interpolate = shape[0] > len(self.raw_pts)
        
        if do_interpolate:
            ts = np.linspace(0, 1., len(self.raw_pts))
            interp_ts = np.linspace(0, 1., shape[0])
            fx = interp1d(ts, self.raw_pts[:, 0], kind='linear')
            fy = interp1d(ts, self.raw_pts[:, 1], kind='linear')
            return jnp.stack([fx(interp_ts), fy(interp_ts)], axis=-1)
        else:
            spacing = len(self.raw_pts) // shape[0]
            if spacing == 0:
                spacing = 1
            indices = np.arange(0, len(self.raw_pts), spacing)
            indices = np.clip(indices, 0, len(self.raw_pts) - 1)
            x = self.raw_pts[indices, 0]
            y = self.raw_pts[indices, 1]
            evaluation = jnp.stack([x, y], axis=-1)
            assert evaluation.shape == shape, f"Expected shape {shape}, but got {evaluation.shape}"
            return evaluation   

class DataFactory:
    
    @staticmethod
    def create(data_type, data_config):
        data_classes = {
            "toy": ToyFunction,
            "ellipse": Ellipse,
            "ellipsoid": Ellipsoid,
            "butterfly": Butterfly, 
        }
        
        if data_type.lower() not in data_classes:
            raise ValueError(f"Data type '{data_type}' is not supported.")
        
        data_class = data_classes[data_type.lower()]
        return data_class(**data_config)
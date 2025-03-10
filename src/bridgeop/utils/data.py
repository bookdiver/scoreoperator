import jax.numpy as jnp
from typing import Tuple, Dict, Any

class BaseData:
    def __init__(self, config: Dict[str, Any], data_klass: Any):
        self.start_point = data_klass(**config['start_point'])
        self.end_point = data_klass(**config['end_point'])

    def eval(self, n_pts: Tuple[int,...]) -> Dict[str, jnp.ndarray]:
        start_points = self.start_point.eval(n_pts)
        end_points = self.end_point.eval(n_pts)
        return {'x0': start_points, 'v': end_points}
        
class Quadratic:
    """ Quadratic function: f(x) = ax^2 + c, a function R->R"""
    a: float
    c: float

    def __init__(self, a: float = 1.0, c: float = 0.0,):
        self.a = a
        self.c = c
    
    def eval(self, n_pts: Tuple[int]) -> jnp.ndarray:
        return jnp.expand_dims(jnp.linspace(-1.0, 1.0, n_pts[0]) ** 2 * self.a + self.c, axis=-1)
    

class Ellipse:
    """ Ellipse shape data, which is treated as a function R->R^2, i.e. parametric equations"""
    a: float
    b: float
    shift_x: float
    shift_y: float

    def __init__(self, a: float = 1.0, b: float = 1.0, shift_x: float = 0.0, shift_y: float = 0.0):
        self.a = a
        self.b = b
        self.shift_x = shift_x
        self.shift_y = shift_y
    
    def eval(self, n_pts: Tuple[int]) -> jnp.ndarray:
        t = jnp.linspace(0, 2*jnp.pi, n_pts[0], endpoint=False)
        x = self.a * jnp.cos(t) + self.shift_x
        y = self.b * jnp.sin(t) + self.shift_y
        return jnp.stack([x, y], axis=1)

    
class Ellipsoid:
    """ Ellipsoid shape data, which is treated as a function R->R^3"""
    a: float
    b: float
    c: float
    shift_x: float
    shift_y: float
    shift_z: float
    
    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 1.0, shift_x: float = 0.0, shift_y: float = 0.0, shift_z: float = 0.0):
        self.a = a
        self.b = b
        self.c = c
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.shift_z = shift_z
    
    def eval(self, n_pts: Tuple[int, int]) -> jnp.ndarray:
        # Calculate the number of points for each dimension
        
        # Generate spherical coordinates
        theta = jnp.linspace(0, jnp.pi, n_pts[0])
        phi = jnp.linspace(0, 2*jnp.pi, n_pts[1])
        
        # Create a meshgrid of theta and phi
        theta, phi = jnp.meshgrid(theta, phi)
        
        # Convert spherical coordinates to Cartesian coordinates
        x = self.a * jnp.sin(theta) * jnp.cos(phi) + self.shift_x
        y = self.b * jnp.sin(theta) * jnp.sin(phi) + self.shift_y
        z = self.c * jnp.cos(theta) + self.shift_z
        
        # Reshape the coordinates into a 2D array of shape (n_pts[0], n_pts[1], 3)
        points = jnp.stack([x, y, z], axis=-1)
        
        return points

    
class DataFactory:
    @staticmethod
    def create(config: Dict[str, Any]) -> 'BaseData':
        data_type = config.get('type', '').lower()
        if data_type == 'quadratic':
            return BaseData(config, Quadratic)
        elif data_type == 'ellipse':
            return BaseData(config, Ellipse)
        elif data_type == 'ellipsoid':
            return BaseData(config, Ellipsoid)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
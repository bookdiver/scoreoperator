from jax.random import PRNGKey

from ..data.synthetic import QuadraticData, CircleData

def test_synthetic_data():
    key = PRNGKey(0)
    quadratic = QuadraticData()
    x = quadratic.eval(n=16, rng_key=key)
    assert x.shape == (16, 1)
    
    circle = CircleData()
    x = circle.eval(n=16, rng_key=None)
    assert x.shape == (16, 2)
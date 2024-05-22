import jax.numpy as jnp
import jax.random as random

from diffrax import (UnsafeBrownianPath, DirectAdjoint, 
                     MultiTerm, ODETerm, 
                     ControlTerm, Euler, SaveAt, diffeqsolve)

from ..models.diffusion.sde import *

def test_heat_sde():
    sde = SDE(
        name = "stochastic_heat",
        sigma = 0.1,
        kappa = 0.2,
        dx = 0.01
    )
    key = random.PRNGKey(0)
    x0 = random.normal(key, (64, 64))

    brownian = UnsafeBrownianPath(shape=(64, 64), key=key)
    terms = MultiTerm(
        ODETerm(lambda t, y, _: sde.f(t, y)),
        ControlTerm(lambda t, y, _: sde.g(t, y), brownian)
    )
    solver = Euler()
    saveat = SaveAt(ts=jnp.arange(0.0, 4.0, 1e-2))
    sol = diffeqsolve(terms, solver, t0=0.0, t1=4.0, dt0=1e-2, saveat=saveat, y0=x0, adjoint=DirectAdjoint())

    xs = sol.ys
    assert xs.shape == (401, 64, 64)
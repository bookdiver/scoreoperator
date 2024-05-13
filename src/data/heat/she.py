import jax.numpy as jnp
from jax import random
from jax import jit
import h5py
from tqdm import tqdm

# Domain settings
dx = 0.2
dt = 0.01
nx, ny = 256, 256  # Grid size
T = 5.0  # Total time
nt = int(T / dt)
record_every = 10
downsample = 4

u0 = jnp.zeros((nx, ny))
u0 = u0.at[0, :].set(10.0)
u0 = u0.at[:, 0].set(10.0)

# Diffusion coefficient
kappa = 1.0
sigma = 0.1

assert dt < dx**2 / (4 * kappa), "Unstable scheme"

# Random key for JAX
key = random.PRNGKey(0)

def laplacian(u):
    """Compute the Laplacian of u."""
    return (jnp.roll(u, 1, axis=0) + jnp.roll(u, -1, axis=0) +
            jnp.roll(u, 1, axis=1) + jnp.roll(u, -1, axis=1) - 4 * u) / dx**2

@jit
def step(u, key):
    """Perform a single time step using Euler-Maruyama method."""
    lap = laplacian(u)
    # Simulate noise (scaled by sqrt(dt))
    noise = random.normal(key, (nx, ny)) * jnp.sqrt(dt)
    # Update the solution
    u_next = u + kappa * lap * dt + sigma * noise
    return u_next, random.split(key)[0]

def solve(u0, key):
    """Solve the 2D stochastic heat equation."""
    us = jnp.empty((nt//record_every + 1, nx//downsample, ny//downsample))
    us = us.at[0].set(u0[::downsample, ::downsample])
    u = u0
    for i in range(nt):
        u, key = step(u, key)
        if (i + 1) % record_every == 0:
            us = us.at[(i + 1)//record_every].set(u[::downsample, ::downsample])
    return us

n_sim = 10_000
file_path = f"../raw/heat_boundary_res{nx//downsample}x{ny//downsample}_{nt//record_every}tsteps.h5"
with h5py.File(file_path, "w") as f:
    max_shape = (None, nt//record_every + 1, nx//downsample, ny//downsample)
    dataset = f.create_dataset("data", shape=(0, nt//record_every + 1, nx//downsample, ny//downsample), maxshape=max_shape, dtype="float32")
    for i in tqdm(range(n_sim)):
        key, _ = random.split(key)
        us = solve(u0, key)
        
        current_end = dataset.shape[0]
        new_shape = (current_end + 1, nt//record_every + 1, nx//downsample, ny//downsample)
        dataset.resize(new_shape)

        dataset[current_end:] = us

# # Plotting the result
# import matplotlib.pyplot as plt

# with h5py.File(file_path, "r") as f:
#     us = f["data"][()]
# fig, axes = plt.subplots(2, 5, figsize=(20, 8))
# axes = axes.ravel()
# for i in range(10):
#     im = axes[i].imshow(us[2, (i+1)*5], cmap='hot')
#     axes[i].axis('off')
#     fig.colorbar(im, ax=axes[i])
# fig.savefig(f"./heat_res{nx//downsample}x{ny//downsample}_{nt//record_every}tsteps2.png")
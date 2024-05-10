from matplotlib.collections import LineCollection
from matplotlib import colormaps
import matplotlib.pyplot as plt
import jax.numpy as jnp

from .util import cpu_run

def close_2d_curve(pts: jnp.ndarray):
    """Close a 2D curve by adding the first point to the end."""
    return jnp.concatenate([pts, pts[:1]], axis=0)

@cpu_run
def plot_1d_function_with_pts(ax, pts, color, marker='o', label='', alpha=1.0):
    """Plot a 1d function with sampled points."""
    ax.plot(pts, color=color, marker=marker, label=label, alpha=alpha)
    return ax

@cpu_run
def plot_2d_function_with_pts(ax, pts, color, marker='o', label=''):
    """Plot a 2d function with sampled points."""
    ax.plot(*close_2d_curve(pts).T, color=color, marker=marker, label=label)
    return ax

@cpu_run
def plot_1d_trajectories(ax, traj, target, cmap_name="viridis", plot_target=True, plot_every=1):
    cmap = colormaps.get_cmap(cmap_name)
    colors = cmap(jnp.linspace(0., 1., traj.shape[0]))
    starting, ending = traj[0], traj[-1]
    plot_1d_function_with_pts(ax, starting, color=colors[0], marker='o', label='Start')
    plot_1d_function_with_pts(ax, ending, color=colors[-1], marker='o', label='End')
    if plot_target:
        plot_1d_function_with_pts(ax, target, color=colors[-1], marker='x', label='Target')

    for i in range(traj.shape[0])[::plot_every]:
        plot_1d_function_with_pts(ax, traj[i], color=colors[i], marker='', alpha=0.5)
    
    # plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    return ax

@cpu_run
def plot_2d_trajectories(ax, traj, target, cmap_name="viridis", plot_target=True, plot_every=1):
    cmap = colormaps.get_cmap(cmap_name)
    colors = cmap(jnp.linspace(0., 1., traj.shape[0]))
    starting, ending = traj[0], traj[-1]
    plot_2d_function_with_pts(ax, starting, color=colors[0], marker='o', label='Start')
    plot_2d_function_with_pts(ax, ending, color=colors[-1], marker='o', label='End')
    if plot_target:
        plot_2d_function_with_pts(ax, target, color=colors[-1], marker='x', label='Target')

    for i in range(traj.shape[1])[::plot_every]:
        x = traj[:, i, 0] 
        points = jnp.array([x, traj[:, i, 1]]).T.reshape(-1, 1, 2)
        segments = jnp.concatenate([points[:-1], points[1:]], axis=1)
        # Create a continuous norm to map from data points to colors,
        norm = plt.Normalize(0.0, 1.0)
        lc = LineCollection(segments, cmap=cmap_name, norm=norm, alpha=0.7)
        # Set the values used for colormapping,
        lc.set_array(jnp.linspace(0., 1., traj.shape[0])),
        lc.set_linewidth(0.7),
        line = (ax.add_collection(lc),)

    # plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    return ax

def plot_trajectories(dim, ax, traj, target, cmap_name="viridis", plot_target=True, plot_every=1):
    """Plot 1d or 2d trajectories."""
    if dim == 1:
        traj = traj.squeeze()
        return plot_1d_trajectories(ax, traj, target, cmap_name, plot_target, plot_every)
    elif dim == 2:
        return plot_2d_trajectories(ax, traj, target, cmap_name, plot_target, plot_every)
    else:
        raise ValueError(f"Invalid dimension {dim}")
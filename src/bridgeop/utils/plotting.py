from matplotlib.collections import LineCollection
from matplotlib import colormaps
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
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
def plot_1d_trajectories(ax, traj, target=None, cmap_name="viridis", plot_every=1):
    cmap = colormaps.get_cmap(cmap_name)
    colors = cmap(jnp.linspace(0., 1., traj.shape[0]))
    starting, ending = traj[0], traj[-1]
    plot_1d_function_with_pts(ax, starting, color=colors[0], marker='o', label='Start')
    plot_1d_function_with_pts(ax, ending, color=colors[-1], marker='o', label='End')
    if target is not None:
        plot_1d_function_with_pts(ax, target, color=colors[-1], marker='x', label='Target')

    for i in range(traj.shape[0])[::plot_every]:
        plot_1d_function_with_pts(ax, traj[i], color=colors[i], marker='', alpha=0.5)
    
    # plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    return ax

@cpu_run
def plot_2d_trajectories(ax, traj, target=None, cmap_name="viridis", plot_every=1):
    cmap = colormaps.get_cmap(cmap_name)
    colors = cmap(jnp.linspace(0., 1., traj.shape[0]))
    starting, ending = traj[0], traj[-1]
    plot_2d_function_with_pts(ax, starting, color=colors[0], marker='o', label='Start')
    plot_2d_function_with_pts(ax, ending, color=colors[-1], marker='o', label='End')
    if target is not None:
        plot_2d_function_with_pts(ax, target, color=colors[-1], marker='*', label='Target')

    for i in range(traj.shape[1]):
        x = traj[:, i, 0] 
        points = jnp.array([x, traj[::plot_every, i, 1]]).T.reshape(-1, 1, 2)
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

@cpu_run
def plot_2d_trajectories_in_3d(ax, traj, target=None, cmap_name="viridis"):
    cmap = colormaps.get_cmap(cmap_name)
    colors = cmap(jnp.linspace(0., 1., traj.shape[0]))
    norm = plt.Normalize(0.0, 1.0)
    starting, ending = traj[0], traj[-1]
    ax.plot(*close_2d_curve(starting).T, zs=0, zdir="z", color=colors[0], marker='o', label='Start', zorder=1)
    ax.plot(*close_2d_curve(ending).T, zs=1, zdir="z", color=colors[-1], marker='o', label='End', zorder=3)
    if target is not None:
        ax.plot(*close_2d_curve(target).T, zs=1, zdir="z", color=colors[-1], marker='*', linestyle="--", label='Target')

    for i in range(traj.shape[1]):
        x_vals = traj[:, i, 0]
        y_vals = traj[:, i, 1]
        z_vals = jnp.linspace(0, 1, traj.shape[0])

        points = jnp.array([x_vals, y_vals, z_vals]).T.reshape(-1, 1, 3)
        segments = jnp.concatenate([points[:-1], points[1:]], axis=1)

        lc = Line3DCollection(segments, cmap=cmap_name, norm=norm, alpha=0.7, zorder=2)
        lc.set_array(jnp.linspace(0., 1., traj.shape[0])),
        lc.set_linewidth(0.7),
        line = (ax.add_collection3d(lc),)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$t$")

    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical")
    cbar.set_label("t")
    return ax

def plot_trajectories(dim, ax, traj, target, cmap_name="viridis", plot_every=1):
    """Plot 1d or 2d trajectories."""
    if dim == 1:
        traj = traj.squeeze()
        return plot_1d_trajectories(ax, traj, target, cmap_name, plot_every)
    elif dim == 2:
        return plot_2d_trajectories(ax, traj, target, cmap_name, plot_every)
    else:
        raise ValueError(f"Invalid dimension {dim}")
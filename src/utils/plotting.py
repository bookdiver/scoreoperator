from matplotlib.collections import LineCollection
from matplotlib import colormaps
import matplotlib.pyplot as plt
import jax.numpy as jnp

def close_2d_curve(pts: jnp.ndarray):
    """Close a 2D curve by adding the first point to the end."""
    return jnp.concatenate([pts, pts[:1]], axis=0)

def plot_shape_with_pts(ax, pts, color, marker='o', label=''):
    """Plot a shape with sampled points."""
    ax.plot(*close_2d_curve(pts).T, color=color, label=label)
    ax.scatter(*pts.T, color=color, marker=marker)
    return ax

def plot_trajectories(ax, traj, target, cmap_name="viridis", plot_target=True):
    cmap = colormaps.get_cmap(cmap_name)
    colors = cmap(jnp.linspace(0., 1., traj.shape[0]))
    starting, ending = traj[0], traj[-1]
    plot_shape_with_pts(ax, starting, color=colors[0], marker='o', label='Start')
    plot_shape_with_pts(ax, ending, color=colors[-1], marker='o', label='End')
    if plot_target:
        plot_shape_with_pts(ax, target, color="g", marker='x', label='Target')

    for i in range(traj.shape[1]):
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
    plt.xlabel("$x$")
    plt.ylabel("$y$")

    return ax
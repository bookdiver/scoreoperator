import jax.numpy as jnp

def close_2d_curve(pts: jnp.ndarray):
    """Close a 2D curve by adding the first point to the end."""
    return jnp.concatenate([pts, pts[:1]], axis=0)


def plot_shape_with_pts(ax, pts, color, marker='o', label=''):
    """Plot a shape with sampled points."""
    ax.plot(*close_2d_curve(pts).T, color=color, label=label)
    ax.scatter(*pts.T, color=color, marker=marker)
    return ax
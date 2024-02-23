# The simplest two moons dataset inspired by scikit-learn's make_moons
import jax.numpy as jnp

class TwoMoons:
    def __init__(self, n_pts):
        self.n_pts_inner = n_pts // 2
        self.n_pts_outer = n_pts - self.n_pts_inner
        self.pts = self.sample()
    
    def sample(self):
        outer_circ_x = jnp.cos(jnp.linspace(0, jnp.pi, self.n_pts_outer))
        outer_circ_y = jnp.sin(jnp.linspace(0, jnp.pi, self.n_pts_outer))
        inner_circ_x = 1 - jnp.cos(jnp.linspace(0, jnp.pi, self.n_pts_inner))
        inner_circ_y = 1 - jnp.sin(jnp.linspace(0, jnp.pi, self.n_pts_inner)) - 0.5
        pts = jnp.vstack([jnp.concatenate([outer_circ_x, inner_circ_x]),
                         jnp.concatenate([outer_circ_y, inner_circ_y])]).T
        return pts

    def draw(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.scatter(self.pts[:self.n_pts_outer, 0], self.pts[:self.n_pts_outer, 1], color='red')
        ax.scatter(self.pts[self.n_pts_outer:, 0], self.pts[self.n_pts_outer:, 1], color='blue')
        return fig
        
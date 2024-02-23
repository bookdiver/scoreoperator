import pytest

from ..data.toys.two_moons import TwoMoons

def test_two_moons():
    n_pts = 100
    two_moons = TwoMoons(n_pts)
    assert two_moons.pts.shape == (n_pts, 2)
    fig = two_moons.draw()
    fig.savefig('./plot_results/two_moons.png')
    assert fig is not None
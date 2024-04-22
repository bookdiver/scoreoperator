import equinox as eqx
from diffrax import VirtualBrownianTree

class ReverseVirtualBrownianTree(VirtualBrownianTree):

    @eqx.filter_jit
    def evaluate(self, t0: float, t1: float=None, left: bool=True, use_levy: bool=False):
        return super().evaluate(1.0-t0, 1.0-t1, left, use_levy)
import abc
from typing import Union, Optional, Tuple, Callable

import jax.numpy as jnp
from jax.random import PRNGKey

class FunctionData(abc.ABC):
    """ Function data base class, used for discrete evaluation of the function. 
    """
    do_dim: int                 # dimension of the domain of the function
    co_dim: int                 # dimension of the codomain of the function

    @abc.abstractmethod
    def _noise_fn(self, rng_key: PRNGKey) -> Callable:
        """ Generate the noise added on the function evaluation to ensure the non-zero measure distribution.

        Args:
            rng_key (PRNGKey): random number key.
            
        Returns:
            Callable: noise function
        """
        pass
    
    @abc.abstractmethod
    def _add_noise(self, data: jnp.ndarray, rng_key: PRNGKey) -> jnp.ndarray:
        """ Method used to add noise on the data.

        Args:
            data (jnp.ndarray): clean function evaluation data.
            rng_key (PRNGKey): random number key.

        Returns:
            jnp.ndarray: noisy function evaluation data.
        """
        pass

    @abc.abstractmethod
    def _eval(self, n: Union[int, Tuple[int]]) -> jnp.ndarray:
        """ Evaluation of the function without noise.

        Args:
            n (Union[int, Tuple[int]]): number of evaluation points, int for 1D domain and tuple for higher dimensions

        Returns:
            jnp.ndarray: function evaluations without noise in an array with shape of (*n, co_dim)
        """
        pass

    def eval(self, n: Union[int, Tuple[int]], rng_key: Optional[PRNGKey] = None) -> jnp.ndarray:
        """ Evaluation of the function.

        Args:
            n (Union[int, Tuple[int]]): number of evaluation points, int for 1D domain and tuple for higher dimensions
            rng_key (Optional[PRNGKey], optional): random number key, if None, then no noise will be added. Defaults to None.

        Returns:
            jnp.ndarray: function evaluations.
        """
        
        if rng_key is None:
            return self._eval(n=n)
        else:
            data = self._eval(n=n)
            return self._add_noise(data=data, rng_key=rng_key)
    
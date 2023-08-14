"""Sums of time-series models."""

import tensorflow as tf
import tensorflow_probability as tfp

from astroparticle.python.transitions.transition import Transition

tfd = tfp.distributions
tfl = tf.linalg


class Sum(Transition):
    """Sum of structural time series components.
    """
    def __init__(self,
                 components,
                 dtype=tf.float32,
                 name="Sum"):

        self.latent_size = sum([c.latent_size for c in components])
        self._component_size = len(components)
        self._components = components
        self.dtype = dtype

    def _default_latent_indicies(self):
        return tf.range(self.latent_size)

    def _get_function(self):

        def _transition_fn(step, x):
            mean = []
            scale = []
            idx_start = 0
            for comp in self._components:
                trans_fn = comp.get_function()
                x_c = x[..., idx_start:idx_start+comp.latent_size]
                trans_dist_c = trans_fn(step, x_c)

                mean.append(trans_dist_c.mean())
                scale.append(trans_dist_c.covariance())

                idx_start += comp.latent_size

            mean = tf.concat(mean, axis=-1)
            scale = tfl.LinearOperatorBlockDiag(
                [tfl.LinearOperatorFullMatrix(s) for s in scale]).to_dense()

            transition_dist = tfd.MultivariateNormalTriL(
                loc=mean, scale_tril=scale)
            return transition_dist

        return _transition_fn

    @property
    def components(self):
        return self._components

    @property
    def component_size(self):
        return self._component_size

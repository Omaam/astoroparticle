"""Constant offset."""

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from partical_xspec.transitions.transition import Transition


class TransitionConstantOffset(Transition):
    def __init__(self, constant_offset, dtype=tf.float32):
        """
        Args:
            constant_offset: float Tensor of shape broadcasting to
                concat([batch_shape, [num_timesteps], [num_latents]])
                specifying a constant value added to the sum of outputs
                from the component models.
        """
        self.constant_offset = constant_offset
        self.num_latents = constant_offset.shape[-1]

        self.dtype = dtype

    def _default_latent_indicies(self):
        latent_indicies = tf.range(self.num_latents)
        return latent_indicies

    def _transition_function(self):
        scale_diag = tf.zeros(self.num_latents, dtype=self.dtype)

        def _transition_fn(i, x):
            means = self.constant_offset[i]
            transition_dist = tfd.MultivariateNormalDiag(
                means, scale_diag=scale_diag)
            return transition_dist

        return _transition_fn

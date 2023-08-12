"""Constant offset."""

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from astroparticle.python.transitions.transition import Transition


class ConstantOffset(Transition):
    def __init__(self, constant_offset, dtype=tf.float32):
        """Constant offset transition class.

        Args:
            constant_offset: float Tensor of shape broadcasting to
                concat([batch_shape, [latent_size]])
                specifying a constant value added to the sum of outputs
                from the component models.
            dtype: The type of an element in the resulting Tensor.
        """
        constant_offset = tf.convert_to_tensor(
            constant_offset, dtype=dtype,
            name="constant_offset")

        self.constant_offset = constant_offset
        self.latent_size = constant_offset.shape[-1]
        self.dtype = dtype

    def _default_latent_indicies(self):
        latent_indicies = tf.range(self.latent_size)
        return latent_indicies

    def _get_function(self):
        means = self.constant_offset
        scale_diag = tf.zeros(self.latent_size, dtype=self.dtype)

        def _transition_fn(i, x):
            transition_dist = tfd.MultivariateNormalDiag(
                means, scale_diag=scale_diag)
            return transition_dist

        return _transition_fn

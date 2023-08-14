"""Constant offset."""

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from astroparticle.python.transitions.transition import Transition


class ConstantOffset(Transition):
    def __init__(self,
                 constant_offset,
                 dtype=tf.float32,
                 name="ConstantOffset"):
        """Constant offset transition class.

        Args:
            constant_offset: float Tensor of shape broadcasting to
                concat([batch_shape, [latent_size]])
                specifying a constant value added to the sum of outputs
                from the component models.
            dtype: The type of an element in the resulting Tensor.
        """
        with tf.name_scope(name):
            constant_offset = tf.convert_to_tensor(
                constant_offset, dtype=dtype,
                name="constant_offset_value")

            self.constant_offset = constant_offset
            self.latent_size = constant_offset.shape[-1]
            self.dtype = dtype

    def _default_latent_indicies(self):
        latent_indicies = tf.range(self.latent_size)
        return latent_indicies

    def _get_function(self):
        def _transition_fn(i, x):
            batch_shape = x.shape[:-1]
            means = tf.broadcast_to(
                self.constant_offset,
                tf.concat([batch_shape, [self.latent_size]], axis=-1)
            )
            scale_diag = tf.zeros(
                tf.concat([batch_shape, [self.latent_size]], axis=-1),
                dtype=self.dtype)

            transition_dist = tfd.MultivariateNormalDiag(
                means, scale_diag=scale_diag)
            return transition_dist

        return _transition_fn

"""Transition class.
"""
import math

import tensorflow as tf

from astroparticle.python.transitions import util as trans_util
from astroparticle.python.experimental.transitions.core \
    import LinearLatentModel


class TrendLatentModel(LinearLatentModel):
    def __init__(self,
                 order,
                 num_dims,
                 dtype=tf.float32,
                 name="Trend"):

        with tf.name_scope(name) as name:

            transition_matrix = self._compute_transition_matrix(
                order, num_dims, dtype)

            self.order = order
            self.dtype = dtype

            super(TrendLatentModel, self).__init__(
                transition_matrix=transition_matrix,
                num_dims=num_dims
            )

    def _default_latent_indices(self, **kwargs):
        return tf.range(self.num_dims * self.order)

    def _compute_transition_matrix(self, order, num_dims, dtype):

        coefficients = tf.constant(
            [math.comb(order, i) for i in range(order)],
            dtype=dtype)
        base_coefficients = coefficients[::-1] * tf.repeat(
            tf.constant([1.0, -1.0], dtype=dtype), order, axis=0)[::2]
        coefficients = tf.convert_to_tensor(
            [tf.linalg.diag(
                tf.repeat(base_coefficients[i], num_dims)
             ) for i in range(order)],
            dtype=dtype
        )
        return trans_util.make_companion_matrix(coefficients)

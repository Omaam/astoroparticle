"""Transition class.
"""
import tensorflow as tf

from astroparticle.python.transitions import util as trans_util
from astroparticle.python.experimental.transitions.core \
    import LinearLatentModel


class VectorAutoregressive(LinearLatentModel):
    def __init__(self,
                 coefficients,
                 dtype=tf.float32,
                 name="VectorAutoregressive"):
        """Build a vector autoregressive model.

        Args:
            coefficients

        Raises:
            ValueError: If coefficient.rank have less than 2.
        """
        with tf.name_scope(name) as name:

            coefficients = tf.convert_to_tensor(
                coefficients, name="coefficients", dtype=dtype)

            if len(coefficients.shape) < 3:
                raise ValueError(
                    "Autoregressive coefficients must have 3 "
                    "dimensions at least.")

            num_dims = coefficients.shape[-1]
            order = coefficients.shape[-3]

            transition_matrix = trans_util.make_companion_matrix(
                coefficients)

            self.coefficients = coefficients
            self.order = order

            super(VectorAutoregressive, self).__init__(
                transition_matrix=transition_matrix,
                num_dims=num_dims
            )

    def _default_latent_indices(self, **kwargs):
        return tf.range(self.num_dims * self.order)

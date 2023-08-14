"""Transition function module.
"""
from tensorflow_probability import distributions as tfd
import tensorflow as tf

from astroparticle.python.transitions import util as trans_util
from astroparticle.python.transitions.transition import Transition


class VectorAutoregressive(Transition):
    """
    """
    def __init__(self,
                 coefficients,
                 noise_covariance,
                 dtype=tf.float32,
                 name="VectorAutoregressive"):
        """
        """
        with tf.name_scope(name):
            coefficients = tf.convert_to_tensor(coefficients, dtype=dtype)
            noise_covariance = tf.convert_to_tensor(
                noise_covariance, dtype=dtype)

            self.coefficients = coefficients
            self.order = coefficients.shape[-3]
            self.latent_size = coefficients.shape[-2]
            self.dtype = dtype

            self.transition_matrix = trans_util.make_companion_matrix(
                coefficients)

            self.transition_noise_cov_chol = tf.linalg.LinearOperatorBlockDiag(
                [tf.linalg.LinearOperatorFullMatrix(
                    tf.linalg.cholesky(noise_covariance)),
                 tf.linalg.LinearOperatorIdentity(
                    (self.order-1)*self.latent_size, dtype=self.dtype)]
            ).to_dense()

    def _default_latent_indicies(self):
        return tf.range(self.latent_size)

    def _get_function(self):

        transition_matrix = self.transition_matrix
        transition_noise_cov_chol = self.transition_noise_cov_chol

        transition_matrix_transpose = tf.linalg.matrix_transpose(
            transition_matrix)

        def _transition_fn(_, x):
            x = tf.convert_to_tensor(x[tf.newaxis, :], self.dtype)
            # TODO: Use `tf.matmul()`, instead of '@'.
            fx = x @ transition_matrix_transpose
            transition_dist = tfd.MultivariateNormalTriL(
                loc=tf.squeeze(fx, axis=0),
                scale_tril=transition_noise_cov_chol)
            return transition_dist

        return _transition_fn

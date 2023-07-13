"""Transition function module.
"""
from tensorflow_probability import distributions as tfd
import tensorflow as tf

from partical_xspec.transition import util as trans_util
from partical_xspec.transition.transition import Transition


class VectorAutoregressiveTransition(Transition):
    """
    """
    def __init__(self, coefficients, noise_covariance, dtype=tf.float32):
        """
        """
        coefficients = tf.convert_to_tensor(coefficients, dtype=dtype)
        self.coefficients = coefficients
        self.order = coefficients.shape[-3]
        self.latent_size = coefficients.shape[-2]

        self.transition_matrix = trans_util.make_companion_matrix(
            coefficients)

        self.transition_noise_cov_chol = tf.linalg.LinearOperatorBlockDiag(
            [tf.linalg.LinearOperatorFullMatrix(
                tf.linalg.cholesky(noise_covariance)),
             tf.linalg.LinearOperatorIdentity(
                (self.order-1)*self.latent_size)]).to_dense()

        self.dtype = dtype

    def _transition_function(self):

        transition_matrix = self.transition_matrix
        transition_noise_cov_chol = self.transition_noise_cov_chol

        transition_matrix_transpose = tf.linalg.matrix_transpose(
            transition_matrix)

        def _transition_fn(_, x):
            x = tf.convert_to_tensor(x[tf.newaxis, :], self.dtype)
            fx = x @ transition_matrix_transpose
            return tfd.MultivariateNormalTriL(
                loc=tf.squeeze(fx, axis=0),
                scale_tril=transition_noise_cov_chol)

        return _transition_fn

    def _proposal_function(self):

        transition_matrix = self.transition_matrix
        transition_noise_cov_chol = self.transition_noise_cov_chol

        transition_matrix_transpose = tf.linalg.matrix_transpose(
            transition_matrix)

        def _proposal_fn(_, x):
            x = tf.convert_to_tensor(x[tf.newaxis, :], self.dtype)
            fx = x @ transition_matrix_transpose
            return tfd.MultivariateNormalTriL(
                loc=tf.squeeze(x-fx, axis=0),
                scale_tril=transition_noise_cov_chol)

        return _proposal_fn


def get_transition_function_var(coefficients,
                                noise_covariance,
                                dtype=tf.float32):

    coefficients = tf.convert_to_tensor(coefficients, dtype=dtype)
    order = coefficients.shape[-3]
    latent_size = coefficients.shape[-2]

    transition_matrix = trans_util.make_companion_matrix(coefficients)
    transition_matrix_transpose = tf.linalg.matrix_transpose(
        transition_matrix)

    eps = tf.constant(0, dtype=dtype)
    transition_noise_cov_chol = tf.linalg.LinearOperatorBlockDiag(
        [tf.linalg.LinearOperatorFullMatrix(
            tf.linalg.cholesky(noise_covariance)),
         tf.linalg.LinearOperatorScaledIdentity(
            (order-1)*latent_size, eps)])

    def _transition_fn(_, x):
        x = tf.convert_to_tensor(x[tf.newaxis, :], dtype)
        fx = x @ transition_matrix_transpose
        return tfd.MultivariateNormalTriL(
            loc=tf.squeeze(fx, axis=0),
            scale_tril=transition_noise_cov_chol.to_dense())

    return _transition_fn


def get_proposal_function_var(coefficients,
                              noise_covariance,
                              dtype=tf.float32):

    coefficients = tf.convert_to_tensor(coefficients, dtype=dtype)
    order = coefficients.shape[-3]
    latent_size = coefficients.shape[-2]

    transition_matrix = trans_util.make_companion_matrix(coefficients)
    transition_matrix_transpose = tf.linalg.matrix_transpose(
        transition_matrix)

    eps = tf.constant(0, dtype=dtype)
    transition_noise_cov_chol = tf.linalg.LinearOperatorBlockDiag(
        [tf.linalg.LinearOperatorFullMatrix(
            tf.linalg.cholesky(noise_covariance)),
         tf.linalg.LinearOperatorScaledIdentity(
            (order-1)*latent_size, eps)])

    def _proposal_function(_, x):
        x = tf.convert_to_tensor(x[tf.newaxis, :], dtype)
        fx = x @ transition_matrix_transpose
        return tfd.MultivariateNormalTriL(
            loc=tf.squeeze(x-fx, axis=0),
            scale_tril=transition_noise_cov_chol.to_dense())

    return _proposal_function

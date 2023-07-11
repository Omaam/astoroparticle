"""Transition function module.
"""
from tensorflow_probability import distributions as tfd
import tensorflow as tf


def make_var_transition_matrix(coefficients):
    """Build transition matrix for an vactor autoregressive StateSpaceModel.

    When applied to a vector of previous values, this matrix computes
    the expected new value (summing the previous states according to the
    autoregressive coefficients) in the top dimension of the state space,
    and moves all previous values down by one dimension, 'forgetting' the
    final (least recent) value. That is, it looks like this:
    ```
    c = c
    var_matrix =
        [ c[0, 0, 0], ..., c[0, 0, M], c[1, 0, 0], ..., c[order, 0, M]
          ...,
          c[0, M, 0], ..., c[0, M, M], c[1, M, 0], ..., c[order, M, M]
          1.,         ..., 0 ,         0.,         ..., 0.
          ...
          0.,         ..., 1.,         0.,         ..., 0.
          ...
          0.,         ..., 0.,         1.,         ..., 0.            ]
    ```
    Args:
        coefficients: float `Tensor` of shape `concat([batch_shape, [order]])`.
    Returns:
        ar_matrix: float `Tensor` with shape `concat([batch_shape,
        [order, order]])`.
    """
    dtype = coefficients.dtype
    coef_shape = coefficients.shape
    batch_shape = coef_shape[:-3]
    order, latent_size = coef_shape[-3], coef_shape[-2]
    top_row = tf.concat(tf.unstack(coefficients, axis=-3), axis=-1)
    remaining_rows = tf.concat([
        tf.eye(latent_size * (order - 1), dtype=dtype,
               batch_shape=batch_shape),
        tf.zeros(tf.concat(
                    [batch_shape, (latent_size * (order - 1), latent_size)],
                    axis=0
                 ),
                 dtype=dtype)
    ], axis=-1)
    var_matrix = tf.concat([top_row, remaining_rows], axis=-2)
    return var_matrix


def get_transition_var(coefficients,
                       noise_covariance,
                       dtype=tf.float32):

    coefficients = tf.convert_to_tensor(coefficients, dtype=dtype)
    order = coefficients.shape[-3]
    latent_size = coefficients.shape[-2]

    transition_matrix = make_var_transition_matrix(coefficients)

    transition_noise_cov_chol = tf.linalg.LinearOperatorBlockDiag(
        [tf.linalg.LinearOperatorFullMatrix(
            tf.linalg.cholesky(noise_covariance)),
         tf.linalg.LinearOperatorZeros((order-1)*latent_size)])

    def _transition_function(_, x):
        x = tf.convert_to_tensor(x[tf.newaxis, :], dtype)
        x_tp1 = x @ transition_matrix

        return tfd.MultivariateNormalTriL(
            loc=tf.squeeze(x-x_tp1, axis=0),
            scale_tril=transition_noise_cov_chol.to_dense())

    return _transition_function

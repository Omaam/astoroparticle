"""Utility module for transitoin function.
"""
import tensorflow as tf


def make_companion_matrix(matrix):
    """Make companion matrix.

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
    dtype = matrix.dtype
    coef_shape = matrix.shape
    batch_shape = coef_shape[:-3]
    order, latent_size = coef_shape[-3], coef_shape[-2]
    top_row = tf.concat(tf.unstack(matrix, axis=-3), axis=-1)
    remaining_rows = tf.concat([
        tf.eye(latent_size * (order - 1), dtype=dtype,
               batch_shape=batch_shape),
        tf.zeros(tf.concat(
                    [batch_shape, (latent_size * (order - 1), latent_size)],
                    axis=0
                 ),
                 dtype=dtype)
    ], axis=-1)
    companion_matrix = tf.concat([top_row, remaining_rows], axis=-2)
    return companion_matrix

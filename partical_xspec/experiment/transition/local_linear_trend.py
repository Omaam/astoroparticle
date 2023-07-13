"""Local linear trend model.
"""
import tensorflow as tf
from tensorflow_probability import distributions as tfd


def get_transition_function_local_linear_trend(
        latent_size,
        level_scale,
        slope_scale,
        dtype=tf.float32):
    """

    observations = [y_1, y_2]'
    latents = [level_scale_1, slope_scale_1, level_scale_2, slope_scale_2]'
    transition_matrix = [[1.0, 1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0  0.0],
                         [0.0, 0.0, 1.0, 1.0],
                         [0.0, 0.0, 0.0, 1.0]]
    observaiton_matrix = [1.0, 0.0, 1.0, 0.0]
    transition_noise ~ N(loc=0., scale=diag([level_scale_1, slope_scale_1,
                                             level_scale_2, slope_scale_2]))
    """
    operator = tf.constant([[1.0, 1.0], [0.0, 1.0]], dtype=dtype)
    transition_matrix = tf.linalg.LinearOperatorBlockDiag(
        [tf.linalg.LinearOperatorFullMatrix(operator)
         for _ in range(latent_size)]).to_dense()
    transition_noise_scale = tf.reshape(
        tf.stack([level_scale, slope_scale], axis=-2), [-1])

    def _transition_function(_, x):
        x = tf.convert_to_tensor(x[tf.newaxis, :], dtype)
        x_tp1 = x @ tf.linalg.matrix_transpose(transition_matrix)
        return tfd.MultivariateNormalDiag(
            loc=tf.squeeze(x-x_tp1, axis=0),
            scale_diag=transition_noise_scale)

    return _transition_function

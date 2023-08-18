"""Core test.
"""
import tensorflow as tf
import tensorflow_probability as tfp

from astroparticle.python.experimental.transitions.core \
    import LinearLatentModel
from astroparticle.python.experimental.transitions.core \
    import SelfOrganizingLatentModel
from astroparticle.python.experimental.transitions.trend \
    import Trend


tfd = tfp.distributions


class _LinearLatentModelTest(tf.test.TestCase):
    def testForward(self):
        transition_matrix = tf.constant(
            [[0.1, 0.0],
             [0.0, 0.1]],
            dtype=self.dtype)
        linear_state_model = LinearLatentModel(
            2, transition_matrix)

        batch_shape = (10,)
        latent_size = transition_matrix.shape[-1]
        particles = tf.ones((*batch_shape, latent_size))

        expected = tf.broadcast_to(
            0.1, (*batch_shape, latent_size))
        actual = linear_state_model.forward(0, particles)
        self.assertAllClose(expected, actual)


class LinearLatentModelTestShape32(
        _LinearLatentModelTest):
    dtype = tf.float32


class LinearLatentModelTestShape64(
        _LinearLatentModelTest):
    dtype = tf.float64


del _LinearLatentModelTest


class _SelfOrganizingLatentModelTest(tf.test.TestCase):
    def testForward(self):

        # Testing parameters.
        ndims = 2
        state_trend_order = 2
        noise_trend_order = 2

        state_model_trend = Trend(
            state_trend_order, ndims, dtype=self.dtype)
        noise_model_trend = Trend(
            noise_trend_order, ndims,
            noise_scale=tf.ones(ndims), dtype=self.dtype)
        transition_dist = tfd.Cauchy
        transition_model = SelfOrganizingLatentModel(transition_dist,
                                                     state_model_trend,
                                                     noise_model_trend)

        batch_shape = (10,)
        latent_size = ndims * (state_trend_order + noise_trend_order)
        particles = tf.broadcast_to(
            tf.constant([0.7, 0.6, 0.3, 0.2, 0.7, 0.6, 0.3, 0.2],
                        dtype=self.dtype),
            (*batch_shape, latent_size))
        transition_dist = transition_model.forward(0, particles)

        actual = transition_dist.mode()

        expected = tf.broadcast_to(
            tf.constant([1.1, 1.0, 0.7, 0.6, 1.1, 1.0, 0.7, 0.6],
                        dtype=self.dtype),
            (*batch_shape, latent_size)
        )
        self.assertAllClose(expected, actual)


class _SelfOrganizingLatentModelTestShape32(_SelfOrganizingLatentModelTest):
    dtype = tf.float32


class _SelfOrganizingLatentModelTestShape64(_SelfOrganizingLatentModelTest):
    dtype = tf.float64


del _SelfOrganizingLatentModelTest


if __name__ == "__main__":
    tf.test.main()

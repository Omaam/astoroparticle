"""Core test.
"""
import tensorflow as tf
import tensorflow_probability as tfp

from astroparticle.python.transitions.core import LinearLatentModel
from astroparticle.python.transitions.core import TransitionModel
from astroparticle.python.transitions.core import TrendLatentModel


tfd = tfp.distributions


class _LinearLatentModelTest(tf.test.TestCase):
    def testForward(self):
        transition_matrix = tf.constant(
            [[0.1, 0.0],
             [0.0, 0.1]],
            dtype=self.dtype)
        linear_state_model = LinearLatentModel(
            transition_matrix, 2)

        batch_shape = (10,)
        latent_size = transition_matrix.shape[-1]
        particles = tf.ones((*batch_shape, latent_size))

        expected = tf.broadcast_to(
            0.1, (*batch_shape, latent_size))
        actual = linear_state_model.forward(0, particles)
        self.assertAllClose(expected, actual)


class LinearLatentModelTestShape32(_LinearLatentModelTest):
    dtype = tf.float32


class LinearLatentModelTestShape64(_LinearLatentModelTest):
    dtype = tf.float64


del _LinearLatentModelTest


class _TransitionModelTest(tf.test.TestCase):
    def testForward(self):

        # Testing parameters.
        state_size = 2
        state_trend_order = 2
        noise_trend_order = 2

        state_model_trend = TrendLatentModel(
            state_trend_order, state_size, dtype=self.dtype)
        noise_model_trend = TrendLatentModel(
            noise_trend_order, state_size, dtype=self.dtype)
        transition_dist = tfd.Cauchy
        transition_model = TransitionModel(transition_dist,
                                           state_model_trend,
                                           noise_model_trend)

        batch_shape = (10,)
        latent_size = state_size * (state_trend_order + noise_trend_order)
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


class _TransitionModelTestShape32(_TransitionModelTest):
    dtype = tf.float32


class _TransitionModelTestShape64(_TransitionModelTest):
    dtype = tf.float64


del _TransitionModelTest


class _TrendLatentModelTest(tf.test.TestCase):
    def testForward(self):
        order = 2
        latent_size = 2
        trend_model = TrendLatentModel(order, latent_size, dtype=self.dtype)

        batch_shape = (10,)
        particles = 0.5 * tf.ones((*batch_shape, order*latent_size))
        particles = tf.broadcast_to(
            0.1 * tf.range(4)[::-1],
            (*batch_shape, order*latent_size))

        expected = tf.broadcast_to(
            [0.5, 0.4, 0.3, 0.2],
            (*batch_shape, order*latent_size))
        actual = trend_model.forward(0, particles)
        self.assertAllClose(expected, actual)


class TrendLatentModelTestShape32(_TrendLatentModelTest):
    dtype = tf.float32


class TrendLatentModelTestShape64(_TrendLatentModelTest):
    dtype = tf.float64


del _TrendLatentModelTest


if __name__ == "__main__":
    tf.test.main()

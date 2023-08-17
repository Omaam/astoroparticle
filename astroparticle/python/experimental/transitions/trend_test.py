"""Trend test.
"""
import tensorflow as tf
import tensorflow_probability as tfp

from astroparticle.python.experimental.transitions.trend \
    import TrendLatentModel


tfd = tfp.distributions


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

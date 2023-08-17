"""Trend test.
"""
import tensorflow as tf
import tensorflow_probability as tfp

from astroparticle.python.experimental.transitions.autoregressive \
    import VectorAutoregressive


tfd = tfp.distributions


class _VectorAutoregressiveTest(tf.test.TestCase):
    def testForward(self):

        coefficients = tf.constant(
            [[[0.4, 0.0],
              [0.0, 0.2]],
             [[-0.1, 0.0],
              [0.0, -0.1]]],
            dtype=self.dtype)
        ndims = coefficients.shape[-1]
        order = coefficients.shape[-3]

        trend_model = VectorAutoregressive(
            coefficients, dtype=self.dtype)

        batch_shape = (10,)
        particles = 0.5 * tf.ones((*batch_shape, order*ndims))

        expected = tf.broadcast_to(
            [0.15, 0.05, 0.5, 0.5],
            (*batch_shape, order*ndims))
        actual = trend_model.forward(0, particles)
        self.assertAllClose(expected, actual)


class VectorAutoregressiveTestShape32(_VectorAutoregressiveTest):
    dtype = tf.float32


class VectorAutoregressiveTestShape64(_VectorAutoregressiveTest):
    dtype = tf.float64


del _VectorAutoregressiveTest


if __name__ == "__main__":
    tf.test.main()

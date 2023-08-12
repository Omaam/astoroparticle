"""Constant offset test."""
import tensorflow as tf

from astroparticle.python.transitions.constant_offset import ConstantOffset


class _ConstantOffsetTest(tf.test.TestCase):

    def testUpdateValue(self):

        # Test conditions.
        batch_shape = [10, 100]
        constants = tf.constant([1.0, 2.0, 3.0], dtype=self.dtype)

        latent_size = constants.shape[-1]
        constant_offset = tf.broadcast_to(constants,
                                          [*batch_shape, latent_size])

        transition = ConstantOffset(constant_offset, dtype=self.dtype)
        transition_fn = transition.get_function()

        # Expect transition_function returns the same values all the time.
        x = tf.broadcast_to(
            tf.random.normal((*batch_shape, latent_size), dtype=self.dtype),
            [*batch_shape, latent_size])
        step = tf.random.uniform([1], maxval=1000, dtype=tf.int32),
        transition_dist = transition_fn(step, x)

        expect = constant_offset
        actual = transition_dist.sample()
        self.assertAllClose(expect, actual)

    def testBatchShape(self):

        # Test conditions.
        batch_shape = [10, 100]
        constants = tf.constant([1.0, 2.0, 3.0], dtype=self.dtype)

        latent_size = constants.shape[-1]
        constant_offset = tf.broadcast_to(constants,
                                          [*batch_shape, latent_size])
        transition = ConstantOffset(constant_offset, dtype=self.dtype)
        transition_fn = transition.get_function()

        x = tf.broadcast_to(
            tf.random.normal((*batch_shape, latent_size), dtype=self.dtype),
            [*batch_shape, latent_size])
        transition_dist = transition_fn(None, x)

        expect = (*batch_shape, latent_size)
        actual = transition_dist.sample().shape
        self.assertAllClose(expect, actual)


class ConstantOffsetTestShape32(_ConstantOffsetTest):
    dtype = tf.float32


class ConstantOffsetTestShape64(_ConstantOffsetTest):
    dtype = tf.float64


del _ConstantOffsetTest  # Don't run tests for the base class.


if __name__ == "__main__":
    tf.test.main()

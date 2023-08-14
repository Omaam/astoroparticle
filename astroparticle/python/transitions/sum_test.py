"""Constant offset test."""
import tensorflow as tf

from astroparticle.python.transitions.sum import Sum
from astroparticle.python.transitions.vector_autoregressive \
    import VectorAutoregressive
from astroparticle.python.transitions.constant_offset import ConstantOffset


class _SumTest(tf.test.TestCase):

    def testUpdatedDistMean(self):

        var = VectorAutoregressive(
            coefficients=[[[0.1, 0.0], [0.0, 0.1]]],
            noise_covariance=[[0.1, 0.0], [0.0, 0.1]],
            dtype=self.dtype)
        const_offset = ConstantOffset(
            constant_offset=[1.0, 2.0],
            dtype=self.dtype)
        sum_model = Sum([var, const_offset], dtype=self.dtype)
        transition_fn = sum_model.get_function()

        # Expect transition_function returns the same values all the time.
        x = tf.ones([4], dtype=self.dtype)
        step = tf.constant(0, dtype=self.dtype)
        transition_dist = transition_fn(step, x)

        expected_mean = [0.1, 0.1, 1.0, 2.0]
        actual_mean = transition_dist.mean()
        self.assertAllClose(expected_mean, actual_mean)

    def testBatchShape(self):
        pass


class SumTestShape32(_SumTest):
    dtype = tf.float32


class SumTestShape64(_SumTest):
    dtype = tf.float64


del _SumTest  # Don't run tests for the base class.


if __name__ == "__main__":
    tf.test.main()

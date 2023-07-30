"""Trend model test.
"""
import unittest

import tensorflow as tf
import numpy as np

from astropf.python.transition.trend import TransitionTrend


class TransitionTrendTest(unittest.TestCase):

    DTYPE = tf.float32

    def test_transtion_matrix(self):
        order_latentsize = [
            [1, 1],
            [1, 2],
            [2, 1],
            [2, 2],
        ]
        expect_means = [
            [0.0],
            [0.0, 0.0],
            [-1.0, 1.0],
            [-1.0, -1.0, 1.0, 1.0]
        ]
        for (order, latent_size), expect_mean in zip(order_latentsize,
                                                     expect_means):
            transtion_dist = self._get_transition_dist(
                order, latent_size, 0.1)
            actual_mean = transtion_dist.mean()
            np.testing.assert_array_equal(expect_mean, actual_mean)

    def _get_transition_dist(
            self, order, latent_size, noise_scale):

        dtype = self.DTYPE
        trans_trend = TransitionTrend(order, latent_size, noise_scale)
        transition_fn = trans_trend.transition_function

        x = tf.concat(
            [tf.ones(latent_size, dtype=dtype),
             tf.zeros((order-1)*latent_size, dtype=dtype)],
            axis=-1
        )
        transition_dist = transition_fn(None, x)
        return transition_dist


if __name__ == "__main__":
    unittest.main()

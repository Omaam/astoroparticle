"""Trend model test.
"""
import unittest

import tensorflow as tf

import astropf.python.transition.local_linear_trend as llt


class TrendModelTest(unittest.TestCase):

    DTYPE = tf.float32

    def test_transtion_matrix(self):
        level_slope_scales = [[0.1, 0.5, 0.2, 1.0]]
        for level_slope_scale in level_slope_scales:
            level_scale = level_slope_scale[:2]
            slope_scale = level_slope_scale[2:]
            transtion_dist = self._get_transition_dist(
                2, level_scale, slope_scale)
            actual_mean = transtion_dist.mean()
            print(actual_mean)

    def _get_transition_dist(self, latent_size, level_scale, slope_scale):

        dtype = self.DTYPE
        transition_fn = llt.get_transition_function_local_linear_trend(
            2, level_scale, slope_scale, dtype=dtype)

        x = tf.constant([1.0, 1.0, 1.0, 1.0])
        transition_dist = transition_fn(None, x)
        return transition_dist


if __name__ == "__main__":
    unittest.main()

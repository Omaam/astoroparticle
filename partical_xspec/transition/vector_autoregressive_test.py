"""Vector autoregressive test.
"""
import unittest

import numpy as np
import tensorflow as tf

import partical_xspec as px
import partical_xspec.transition.vector_autoregressive as var


class VectorAutoregressiveTest(unittest.TestCase):

    DTYPE = tf.float32

    def test_transition_matrix(self):
        coefficients = tf.convert_to_tensor(
            np.array([[[0.1, 0.0],
                       [0.0, 0.1]],
                      [[0.1, 0.0],
                       [0.0, 0.1]]]),
            dtype=self.DTYPE)

        actual = var.make_var_transition_matrix(coefficients)
        expect = tf.convert_to_tensor(
            np.array([[0.1, 0.0, 0.1, 0.0],
                      [0.0, 0.1, 0.0, 0.1],
                      [1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0]]),
            dtype=self.DTYPE)

        np.testing.assert_array_equal(expect, actual)

    def test_update_latents(self):
        dtype = self.DTYPE
        coefficients = tf.convert_to_tensor(
            np.array([[[0.1, 0.0],
                       [0.0, 0.1]],
                      [[0.1, 0.0],
                       [0.0, 0.1]]]),
            dtype=dtype)
        noise_covariance = tf.convert_to_tensor(
            np.array([[0.01, 0.00],
                      [0.00, 0.01]]),
            dtype=dtype)
        transition_fn = px.get_transition_function_varmodel(
            coefficients, noise_covariance)

        x = tf.constant([1.0, 1.0, 0.0, 0.0])
        transition_dist = transition_fn(None, x)
        self.assertEqual(4, transition_dist.event_shape)


if __name__ == "__main__":
    unittest.main()

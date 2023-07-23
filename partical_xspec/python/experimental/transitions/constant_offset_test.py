"""Constant offset test."""
import tensorflow as tf

from partical_xspec.python.transition.experimental.constant_offset \
    import TransitionConstantOffset


class ConstantOffsetTest(tf.test.TestCase):

    DTYPE = tf.float32

    def test_output_values(self):

        trans_constant_offset = self.get_test_case_00()

        transition_fn = trans_constant_offset.transition_function
        inputs = 0, tf.constant([0.0, 0.0])
        transition_dist = transition_fn(*inputs)

        expect = tf.constant([1.0, 1.0], dtype=self.DTYPE)
        actual = transition_dist.sample()
        self.assertAllClose(expect, actual)

    def get_test_case_00(self):
        num_timesteps = 10
        num_latents = 2
        constant_offsets = tf.ones((num_timesteps, num_latents),
                                   dtype=tf.float32)
        return TransitionConstantOffset(constant_offsets)


if __name__ == "__main__":
    tf.test.main()

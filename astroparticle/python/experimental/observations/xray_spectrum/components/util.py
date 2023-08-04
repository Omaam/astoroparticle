"""Component util module.
"""
import tensorflow as tf


def compute_section_trapezoidal(energy_intervals, function):
    """Compute section by trapezoidal rule.

    Args:
        energy_intervals: float tensor of shape
            `concat (batch_shape, num_energies, 2)` specifying
            energy intervals.
        function: callable specifying function to compute section.
    """
    width = energy_intervals[..., 1] - energy_intervals[..., 0]
    heights_div2 = tf.reduce_sum(function(energy_intervals), axis=-1) / 2
    return tf.multiply(width[tf.newaxis, :], heights_div2,
                       name="trapezoidal_section")

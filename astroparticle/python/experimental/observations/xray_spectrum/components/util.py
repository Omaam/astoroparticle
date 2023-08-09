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


def compute_section_trapezoidal2(energy_edges, function):
    """Compute section by trapezoidal rule.

    Args:
        energy_edges: float tensor of shape
            `concat (batch_shape, num_edges)` specifying energy edges.
        function: callable specifying function to compute section.
    """
    widths = energy_edges[..., 1:] - energy_edges[..., :-1]
    heights = function(energy_edges[..., :-1]) + function(
        energy_edges[..., 1:])

    mean_widths = tf.reduce_mean(widths)
    if mean_widths >= 1.:
        import warnings
        warnings.warn(
            "mean of energy intervals is greater than 1.0, which "
            "may cause flux calculation poor."
        )

    return widths * heights / 2

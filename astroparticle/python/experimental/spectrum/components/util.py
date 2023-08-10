"""Component util module.
"""


def compute_section_trapezoidal(energy_edges, function):
    """Compute section by trapezoidal rule.

    Args:
        energy_edges: float tensor of shape
            `concat (batch_shape, num_edges)` specifying energy edges.
        function: callable specifying function to compute section.
    """
    widths = energy_edges[..., 1:] - energy_edges[..., :-1]
    heights = function(energy_edges[..., :-1]) + function(
        energy_edges[..., 1:])
    return widths * heights / 2

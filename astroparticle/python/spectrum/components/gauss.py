"""Energy spectrum component model module.
"""
import math

import tensorflow as tf

from astroparticle.python.experimental.spectrum.components.physical_component \
    import PhysicalComponent
from astroparticle.python.experimental.spectrum.components \
    import util as comp_util


def gauss(energy, line_energy, line_width, norm):
    """Compute probability density for gauss function.

    A(E) = K * 1/(sigma * sqrt(2*pi)) exp(-(E-El)^2)/ 2*sigma^2)
    """
    return norm / line_width / tf.sqrt(2*math.pi) * tf.exp(
        -(energy - line_energy)**2 / 2 / line_width**2)


class Gauss(PhysicalComponent):
    def __init__(self,
                 energy_edges,
                 energy_line=6.4,
                 energy_width=0.1,
                 normalization=1.0,
                 name="normalization"):
        with tf.name_scope(name) as name:

            self.energy_line = tf.convert_to_tensor(energy_line)
            self.energy_width = tf.convert_to_tensor(energy_width)
            self.normalization = tf.convert_to_tensor(normalization)

            super(Gauss, self).__init__(
                energy_edges_input=energy_edges,
                energy_edges_output=energy_edges
            )

    def _forward(self, flux):
        """Forward to calculate flux.
        """
        # TODO: Many uses of `tf.newaxis` make a mess.
        # Find another tider way.
        energy_edges = self.energy_edges_input[tf.newaxis, ...]
        energy_line = self.energy_line[:, tf.newaxis]
        energy_width = self.energy_width[:, tf.newaxis]
        norm = self.normalization[:, tf.newaxis]

        def _gauss_with_param(energy_edges):
            return gauss(energy_edges, energy_line, energy_width, norm)

        new_flux = comp_util.compute_section_trapezoidal(
            energy_edges, _gauss_with_param)

        flux = flux + new_flux
        return flux

    def set_parameter(self, x):
        x = tf.unstack(x, axis=-1)
        self.energy_line = x[0]
        self.energy_width = x[1]
        self.normalization = x[2]

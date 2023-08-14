"""Energy spectrum component model module.
"""
import tensorflow as tf

from astroparticle.python.spectrum.components.physical_component \
    import PhysicalComponent
from astroparticle.python.spectrum.components import util as comp_util


class PowerLaw(PhysicalComponent):
    def __init__(self,
                 energy_edges,
                 photon_index=1.0,
                 normalization=1.0,
                 dtype=tf.float32, name="powerlaw"):

        with tf.name_scope(name) as name:
            self.photon_index = tf.convert_to_tensor(
                photon_index, dtype=dtype)
            self.normalization = tf.convert_to_tensor(
                normalization, dtype=dtype)
            super(PowerLaw, self).__init__(
                energy_edges_input=energy_edges,
                energy_edges_output=energy_edges)

            self.dtype = dtype
            self._parameter_size = 2

    def _forward(self, flux):
        """Forward to calculate flux.
        """
        energy_edges = self.energy_edges_input
        photon_index = self.photon_index[..., tf.newaxis]
        norm = self.normalization[..., tf.newaxis]

        def _powerlaw(energies):
            return tf.math.pow(energies, -photon_index)

        new_flux = norm * comp_util.compute_section_trapezoidal(
                energy_edges, _powerlaw)

        flux = flux + new_flux
        return flux

    def _set_parameter(self, x):
        x = tf.unstack(x, axis=-1)
        self.photon_index = x[0]
        self.normalization = x[1]

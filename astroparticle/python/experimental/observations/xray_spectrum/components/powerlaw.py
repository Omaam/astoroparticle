"""Energy spectrum component model module.
"""
import tensorflow as tf

from astroparticle.python.experimental.observations.\
    xray_spectrum.components.physical_component import PhysicalComponent
from astroparticle.python.experimental.observations.\
    xray_spectrum.components import util as comp_util


class PowerLaw(PhysicalComponent):
    def __init__(self,
                 energy_intervals_input,
                 photon_index=1.0,
                 normalization=1.0,
                 dtype=tf.float32, name="powerlaw"):

        with tf.name_scope(name) as name:
            self._energy_intervals_input = energy_intervals_input
            self.photon_index = tf.convert_to_tensor(
                photon_index, dtype=dtype)
            self.normalization = tf.convert_to_tensor(
                normalization, dtype=dtype)
            super(PowerLaw, self).__init__(
                energy_intervals_input=energy_intervals_input,
                energy_intervals_output=energy_intervals_input)

    def _forward(self, flux):
        """Forward to calculate flux.
        """
        # TODO: Many uses of `tf.newaxis` make a mess.
        # Find another tider way.
        energy_intervals = self.energy_intervals_input
        photon_index = self.photon_index[:, tf.newaxis, tf.newaxis]
        norm = self.normalization[:, tf.newaxis]
        print("enegy_intervals: {}".format(energy_intervals.shape))
        print("norm: {}".format(norm.shape))

        def _powerlaw(energies):
            return tf.math.pow(energies, -photon_index)

        new_flux = norm * comp_util.compute_section_trapezoidal(
                energy_intervals, _powerlaw)

        flux = flux + new_flux
        return flux

    def _set_parameter(self, x):
        x = tf.unstack(x, axis=-1)
        self.photon_index = x[0]
        self.normalization = x[1]

"""Energy spectrum component model module.
"""
import tensorflow as tf

from astroparticle.python.experimental.observations.xray_spectrum.core \
    import XraySpectrum


class PowerLaw(XraySpectrum):
    def __init__(self, energy_intervals,  photon_index, normalization,
                 dtype=tf.float32, name="powerlaw"):

        with tf.name_scope(name) as name:
            self._energy_intervals = energy_intervals
            self.photon_index = tf.convert_to_tensor(
                photon_index, dtype=dtype)
            self.normalization = tf.convert_to_tensor(
                normalization, dtype=dtype)
            super(PowerLaw, self).__init__(
                dtype=dtype)

    def _forward(self, flux):
        """Forward to calculate flux.

        TODO:
            Powerlaw in xspec should be referenced.
        """
        photon_index = self.photon_index
        normalization = self.normalization

        energy_centers = tf.reduce_mean(self.energy_intervals_input, axis=1)

        flux = flux + normalization[:, tf.newaxis] * tf.math.pow(
            energy_centers, -photon_index[:, tf.newaxis])

        return flux

    @property
    def _energy_intervals_input(self):
        return self._energy_intervals

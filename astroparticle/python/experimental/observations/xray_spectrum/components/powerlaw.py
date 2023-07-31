"""Energy spectrum component model module.
"""
import tensorflow as tf

from astroparticle.python.experimental.observations.xray_spectrum.core import XraySpectrum


class PowerLaw(XraySpectrum):
    def __init__(self, energy_edges,  photon_index, normalization,
                 name="powerlaw"):

        with tf.name_scope(name) as name:

            self.photon_index = tf.Variable(photon_index)
            self.normalization = tf.Variable(normalization)

            super(PowerLaw, self).__init__(
                energy_edges,
                num_params=2)

    def _forward(self, flux):
        """Forward to calculate flux.

        TODO:
            Powerlaw in xspec should be referenced.
        """
        photon_index = self.photon_index
        normalization = self.normalization

        energy_centers = tf.reduce_mean(self.energy_edges, axis=1)

        flux = flux + normalization[:, tf.newaxis] * tf.math.pow(
            energy_centers, -photon_index[:, tf.newaxis])

        return flux

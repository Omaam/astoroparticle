"""Energy spectrum component model module.
"""
import tensorflow as tf
import tensorflow_probability as tfp

from astroparticle.python.experimental.observations.xray_spectrum.core import XraySpectrum


class Rebin(XraySpectrum):
    def __init__(self, energy_splits_old, energy_splits_new):
        self.energy_splits_old = energy_splits_old
        self.energy_splits_new = energy_splits_new

    def _forward(self, flux):

        energy_splits_old = self.energy_splits_old
        energies_old = (energy_splits_old[1:] + energy_splits_old[:-1]) / 2

        segment_ids = tfp.stats.find_bins(energies_old,
                                          self.energy_splits_new,
                                          dtype=tf.float32)

        # Only handle flux inside the range.
        is_nan = tf.math.logical_not(tf.math.is_nan(segment_ids))
        segment_ids = tf.cast(segment_ids[is_nan], tf.int32)
        flux = flux[..., is_nan]

        binned_flux = tf.map_fn(
            lambda f: tf.math.segment_sum(f, segment_ids), flux)
        return binned_flux

    @property
    def _energies(self):
        energy_splits = self.energy_splits_new
        return (energy_splits[1:] + energy_splits[:-1]) / 2

    @property
    def _energy_edges(self):
        energy_splits_new = self.energy_splits_new
        return tf.concat(
            [energy_splits_new[:-1][:, tf.newaxis],
             energy_splits_new[1:][:, tf.newaxis]],
            axis=1)

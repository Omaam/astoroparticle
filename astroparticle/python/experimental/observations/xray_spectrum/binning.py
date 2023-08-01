"""Energy spectrum component model module.
"""
import tensorflow as tf
import tensorflow_probability as tfp

from astroparticle.python.experimental.observations.\
    xray_spectrum.xray_spectrum import XraySpectrum


class Rebin(XraySpectrum):
    def __init__(self,
                 energy_intervals_input,
                 energy_intervals_output):
        super(Rebin, self).__init__(
            energy_intervals_input,
            energy_intervals_output)

    def _forward(self, flux):

        if flux.shape[-1] != self.energy_intervals_input.shape[-2]:
            raise ValueError()

        energy_centers_input = tf.reduce_mean(
            self.energy_intervals_input,
            axis=-1)

        energy_edges_output = self._convert_intervals_to_edges(
            self.energy_intervals_output)
        segment_ids = tfp.stats.find_bins(energy_centers_input,
                                          energy_edges_output,
                                          dtype=tf.float32)

        # Only handle flux inside the range.
        is_nan = tf.math.logical_not(tf.math.is_nan(segment_ids))
        segment_ids = tf.cast(segment_ids[is_nan], tf.int32)
        flux = flux[..., is_nan]

        binned_flux = tf.map_fn(
            lambda f: tf.math.segment_sum(f, segment_ids), flux)

        return binned_flux

    def _convert_intervals_to_edges(self, intervals):
        return tf.concat([intervals[:, 0], [intervals[-1, 1]]], axis=-1)

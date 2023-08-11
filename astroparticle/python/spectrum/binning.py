"""Energy spectrum component model module.
"""
import tensorflow as tf
import tensorflow_probability as tfp

from astroparticle.python.spectrum.spectrum import Spectrum


class Rebin(Spectrum):
    def __init__(self,
                 energy_edges_input,
                 energy_edges_output):
        super(Rebin, self).__init__(
            energy_edges_input,
            energy_edges_output)

    def _forward(self, flux):

        dtype = flux.dtype

        energy_centers_input = \
            self.energy_edges_input[..., :-1] + \
            self.energy_edges_input[..., 1:]

        segment_ids = tfp.stats.find_bins(energy_centers_input,
                                          self.energy_edges_output,
                                          dtype=dtype)

        # Only handle flux inside the range.
        is_nan = tf.math.logical_not(tf.math.is_nan(segment_ids))
        segment_ids = tf.cast(segment_ids[is_nan], tf.int32)
        flux = flux[..., is_nan]

        binned_flux = tf.map_fn(
            lambda f: tf.math.segment_sum(f, segment_ids), flux)

        return binned_flux

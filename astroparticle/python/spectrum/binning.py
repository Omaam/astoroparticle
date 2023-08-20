"""Energy spectrum component model module.
"""
import tensorflow as tf
import tensorflow_probability as tfp

from astroparticle.python.spectrum.spectrum import Spectrum


class Rebin(Spectrum):
    def __init__(self,
                 energy_edges_input,
                 energy_edges_output):

        energy_edges_input = tf.convert_to_tensor(energy_edges_input)
        energy_edges_output = tf.convert_to_tensor(energy_edges_output)

        super(Rebin, self).__init__(
            energy_edges_input,
            energy_edges_output)

    def _forward(self, flux):

        dtype = flux.dtype

        energy_centers_input = (self.energy_edges_input[..., :-1] +
                                self.energy_edges_input[..., 1:]) / 2

        segment_ids = tfp.stats.find_bins(energy_centers_input,
                                          self.energy_edges_output,
                                          dtype=dtype)

        # Only process fluxes inside the range.
        is_nan = tf.math.logical_not(tf.math.is_nan(segment_ids))
        segment_ids = tf.cast(segment_ids[is_nan], tf.int32)
        flux = flux[..., is_nan]

        # This raises error when using XLA operation.
        binned_flux = tf.linalg.matrix_transpose(
            tf.math.segment_sum(tf.linalg.matrix_transpose(flux),
                                segment_ids)
        )
        # This does not raises error, but it seems that this runs
        # slower than above.
        # num_segments = self.energy_edges_output.shape[-1] - 1
        # binned_flux = tf.linalg.matrix_transpose(
        #     tf.math.unsorted_segment_sum(
        #         tf.linalg.matrix_transpose(flux),
        #         segment_ids,
        #         num_segments=num_segments)
        # )

        return binned_flux

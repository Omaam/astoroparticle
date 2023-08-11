"""Experiment for response function.
"""
import tensorflow as tf
from astropy.io import fits

from astroparticle.python.spectrum.spectrum import Spectrum


class DetectorResponseModel(tf.Module):

    def _get_value_from_file(self, filepath, idx_hdul, column_name):
        with fits.open(filepath) as hdul:
            value = hdul[idx_hdul].data.field(column_name)
        return value

    def _get_header_from_file(self, filepath, idx_hdul, name):
        with fits.open(filepath) as hdul:
            value = hdul[idx_hdul].header[name]
        return value


class AncillaryResponseModel(Spectrum, DetectorResponseModel):
    def __init__(self, filepath, dtype=tf.float32,
                 name="ancillary_response"):
        with tf.name_scope(name) as name:
            self._filepath = filepath
            self._ancillary_response = self._get_ancillary_response(dtype)
            self.dtype = dtype
            super(AncillaryResponseModel, self).__init__(
                self._get_energy_edges_input(dtype),
                self._get_energy_edges_output(dtype))

    def _forward(self, x):
        return tf.math.multiply(self._ancillary_response, x)

    def _get_ancillary_response(self, dtype=tf.float32):
        ancillary_response = tf.convert_to_tensor(
            self._get_value_from_file(self._filepath, 1, "SPECRESP"),
            dtype=dtype)
        return ancillary_response

    def _get_energy_edges_input(self, dtype=tf.float32):
        energy_lows = tf.convert_to_tensor(
            self._get_value_from_file(self._filepath, 1, "ENERG_LO"),
            dtype=self.dtype)
        energy_higs = tf.convert_to_tensor(
            self._get_value_from_file(self._filepath, 1, "ENERG_HI"),
            dtype=self.dtype)
        energy_edges = tf.concat([energy_lows, [energy_higs[-1]]], axis=-1)
        return energy_edges

    def _get_energy_edges_output(self, dtype=tf.float32):
        return self._get_energy_edges_input(dtype)


class ResponseMatrixModel(Spectrum, DetectorResponseModel):
    """
    """
    def __init__(self, filepath, dtype=tf.float32, name="response_matrix"):
        with tf.name_scope(name) as name:
            self._filepath = filepath
            self._response_matrix = self._get_response_matrix(dtype)
            self.dtype = dtype
            super(ResponseMatrixModel, self).__init__(
                self._get_energy_edges_input(dtype),
                self._get_energy_edges_output(dtype))

    def _forward(self, x):
        return tf.matmul(x, self._response_matrix)

    def _get_energy_edges_input(self, dtype=tf.float32):
        energy_lows = tf.convert_to_tensor(
            self._get_value_from_file(self._filepath, 2, "ENERG_LO"),
            dtype=self.dtype)
        energy_higs = tf.convert_to_tensor(
            self._get_value_from_file(self._filepath, 2, "ENERG_HI"),
            dtype=self.dtype)
        energy_edges = tf.concat([energy_lows, [energy_higs[-1]]], axis=-1)
        return energy_edges

    def _get_energy_edges_output(self, dtype=tf.float32):
        energy_lows = tf.convert_to_tensor(
            self._get_value_from_file(self._filepath, 1, "E_MIN"),
            dtype=self.dtype)
        energy_higs = tf.convert_to_tensor(
            self._get_value_from_file(self._filepath, 1, "E_MAX"),
            dtype=self.dtype)
        energy_edges = tf.concat([energy_lows, [energy_higs[-1]]], axis=-1)
        return energy_edges

    def _get_response_matrix(self, dtype=tf.float32):
        num_channel_output = tf.cast(
            self._get_header_from_file(self._filepath, 2, "DETCHANS"),
            dtype=tf.int32)
        raw_response_matrix = tf.ragged.constant(
            self._get_value_from_file(self._filepath, 2, "MATRIX"),
            dtype=dtype).to_tensor()
        num_left_paddings = tf.subtract(num_channel_output,
                                        raw_response_matrix.shape[-1])

        padded_response_matrix = tf.pad(
            raw_response_matrix,
            paddings=tf.convert_to_tensor([[0, 0], [0, num_left_paddings]],
                                          dtype=tf.int32)
        )

        first_channels = tf.convert_to_tensor(
            self._get_value_from_file(self._filepath, 2, "F_CHAN"),
            dtype=tf.int32)

        response_matrix = tf.map_fn(
            lambda x: tf.roll(x[0], x[1], axis=-1, name="roll"),
            elems=[padded_response_matrix, first_channels],
            fn_output_signature=padded_response_matrix.dtype
        )
        return response_matrix


class CustumResponseModel(DetectorResponseModel):

    def __init__(self, rmf_path, arf_path, dtype=tf.float32):
        self.response_matrix = ResponseMatrixModel(
            rmf_path, dtype=dtype)
        self.ancillary_response = AncillaryResponseModel(
            arf_path, dtype=dtype)

    def _forward(self, x):
        y = self.ancillary_response.forward(x)
        y = self.response_matrix.forward(y)
        return y

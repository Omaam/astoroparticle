"""Experiment for response function.

TODO:
    * use `tf.name_scope`.
"""
from astropf.python.experimental.observations.xray_spectrum.core import XraySpectrum
import tensorflow as tf
from astropy.io import fits


class DetectorResponseModel(XraySpectrum):
    def __init__(self, filepath, dtype=tf.float32):
        self.filepath = filepath
        self.dtype = dtype

    def _get_value_from_file(self, idx_hdul, column_name):
        with fits.open(self.filepath) as hdul:
            value = hdul[idx_hdul].data.field(column_name)
        return value

    def _inverse(self, x):
        """Subclass implementation for `inverse` public function."""
        raise NotImplementedError("forward not implemented.")

    def inverse(self, x):
        return self._inverse(x)

    @property
    def response(self):
        return self._response


class AncillaryResponseModel(DetectorResponseModel):
    def __init__(self, filepath, dtype=tf.float32,
                 name="ancillary_response"):
        with tf.name_scope(name) as name:
            super(AncillaryResponseModel, self).__init__(
                filepath=filepath,
                dtype=dtype)
            self._ancillary_response = self._get_ancillary_response()

    def _forward(self, x):
        return tf.math.multiply(self.response, x)

    def _get_ancillary_response(self):
        ancillary_response = tf.convert_to_tensor(
            self._get_value_from_file(1, "SPECRESP"))
        return ancillary_response

    @property
    def _energy_edges(self):
        return tf.concat(
            [self._get_value_from_file(1, "ENERG_LO")[:, tf.newaxis],
             self._get_value_from_file(1, "ENERG_HI")[:, tf.newaxis]],
            axis=1)

    @property
    def _response(self):
        return self._ancillary_response


class ResponseMatrixModel(DetectorResponseModel):
    def __init__(self, filepath, dtype=tf.float32, name="response_matrix"):
        with tf.name_scope(name) as name:

            super(ResponseMatrixModel, self).__init__(
                filepath=filepath,
                dtype=dtype)

            self._response_matrix = self._get_response_matrix()
            self.num_groups = self._response_matrix.shape[0]
            self.num_channels = self._response_matrix.shape[1]

    def _forward(self, x):
        return tf.matmul(x, self._response)

    def _get_response_matrix(self):
        response_matrix = tf.ragged.constant(
            self._get_value_from_file(2, "MATRIX"),
            dtype=self.dtype
        ).to_tensor()
        return response_matrix

    @property
    def _energy_edges(self):
        energy_lows = tf.convert_to_tensor(
            self._get_value_from_file(2, "ENERG_LO")[:, tf.newaxis],
            dtype=self.dtype)
        energy_highs = tf.convert_to_tensor(
            self._get_value_from_file(2, "ENERG_HI")[:, tf.newaxis],
            dtype=self.dtype)
        return tf.concat([energy_lows, energy_highs], axis=1)

    @property
    def _response(self):
        return self._response_matrix


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

    @property
    def _response(self):
        return [self.response_matrix.response,
                self.ancillary_response.response]

    @property
    def _energy_edges(self):
        return self.response_matrix.energy_edges


def main():

    spectrum_processer = CustumResponseModel(
        "data/nixtiref20170601v003.rmf",
        "data/nixtiaveonaxis20170601v005.arf")

    # Usually, this obtain from an energy spectrum model.
    # However, we probide this with poisson noise.
    counts_raw_model = tf.random.poisson(
        (10, 3451), lam=20, seed=123)
    print("input shape: {}".format(counts_raw_model.shape))

    # in cts/s/channel
    counts_obs_model = spectrum_processer.forward(counts_raw_model)
    print("output shape: {}".format(counts_obs_model.shape))


if __name__ == "__main__":
    main()

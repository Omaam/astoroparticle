"""
"""
import os

import tensorflow as tf

from astroparticle.python.experimental.spectrum.spectrum import Spectrum
from astroparticle.python.experimental.spectrum.response \
    import AncillaryResponseModel
from astroparticle.python.experimental.spectrum.response \
    import ResponseMatrixModel


class ResponseNewtonDetectorName(Spectrum):
    def __init__(self):
        raise NotImplementedError()


class ResponseNicerXti(Spectrum):

    def __init__(self, dtype=tf.float32):

        file_dir = os.path.dirname(__file__)
        self.response_matrix = ResponseMatrixModel(
            os.path.join(file_dir, "data/nixtiref20170601v003.rmf"),
            dtype=dtype)
        self.ancillary_response = AncillaryResponseModel(
            os.path.join(file_dir, "data/nixtiaveonaxis20170601v005.arf"),
            dtype=dtype)
        [
         self._energy_edges_input,
         self._energy_edges_output
        ] = [
         self.ancillary_response._energy_edges_input,
         self.response_matrix._energy_edges_output
        ]

    def _forward(self, flux):
        flux = self.ancillary_response.forward(flux)
        flux = self.response_matrix.forward(flux)
        return flux


class ResponseNustarDetectorName(Spectrum):
    def __init__(self):
        raise NotImplementedError()


class ResponseRxteDetectorName(Spectrum):
    def __init__(self):
        raise NotImplementedError()


class ResponseXrismResolve(Spectrum):
    def __init__(self):
        raise NotImplementedError()

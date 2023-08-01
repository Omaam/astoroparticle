"""
"""
import os

import tensorflow as tf

from astroparticle.python.experimental.observations.\
    xray_spectrum.xray_spectrum import XraySpectrum
from astroparticle.python.experimental.observations.xray_spectrum.response \
    import AncillaryResponseModel
from astroparticle.python.experimental.observations.xray_spectrum.response \
    import ResponseMatrixModel


class ResponseNewtonDetectorName(XraySpectrum):
    def __init__(self):
        raise NotImplementedError()


class ResponseNicerXti(XraySpectrum):

    def __init__(self, dtype=tf.float32):
        self.response_matrix = ResponseMatrixModel(
            os.path.join(
                os.path.dirname(__file__),
                "data/nixtiref20170601v003.rmf")
        )
        self.ancillary_response = AncillaryResponseModel(
            os.path.join(
                os.path.dirname(__file__),
                "data/nixtiaveonaxis20170601v005.arf")
        )
        [
         self._energy_intervals_input,
         self._energy_intervals_output
        ] = [
         self.ancillary_response._energy_intervals_input,
         self.response_matrix._energy_intervals_output
        ]

    def _forward(self, x):
        y = self.ancillary_response.forward(x)
        y = self.response_matrix.forward(y)
        return y


class ResponseNustarDetectorName(XraySpectrum):
    def __init__(self):
        raise NotImplementedError()


class ResponseRxteDetectorName(XraySpectrum):
    def __init__(self):
        raise NotImplementedError()


class ResponseXrismResolve(XraySpectrum):
    def __init__(self):
        raise NotImplementedError()

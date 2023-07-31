"""
"""
import os

import tensorflow as tf

from astroparticle.python.experimental.observations.xray_spectrum.response import AncillaryResponseModel
from astroparticle.python.experimental.observations.xray_spectrum.response import DetectorResponseModel
from astroparticle.python.experimental.observations.xray_spectrum.response import ResponseMatrixModel


class ResponseNewtonDetectorName(DetectorResponseModel):
    def __init__(self):
        raise NotImplementedError()


class ResponseNicerXti(DetectorResponseModel):

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
        self.dtype = dtype

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


class ResponseNustarDetectorName(DetectorResponseModel):
    def __init__(self):
        raise NotImplementedError()


class ResponseRxteDetectorName(DetectorResponseModel):
    def __init__(self):
        raise NotImplementedError()


class ResponseXrismResolve(DetectorResponseModel):
    def __init__(self):
        raise NotImplementedError()

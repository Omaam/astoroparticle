"""Physical component.
"""
from astroparticle.python.experimental.observations.xray_spectrum.\
    xray_spectrum import XraySpectrum


class PhysicalComponent(XraySpectrum):

    def _set_parameter(self, x):
        raise NotImplementedError(
            "set_parameter not implemented")

    def set_parameter(self, x):
        self._set_parameter(x)

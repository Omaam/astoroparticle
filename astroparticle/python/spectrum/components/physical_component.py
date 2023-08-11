"""Physical component.
"""
from astroparticle.python.spectrum.spectrum import Spectrum


class PhysicalComponent(Spectrum):

    def _set_parameter(self, x):
        raise NotImplementedError(
            "set_parameter not implemented")

    def set_parameter(self, x):
        self._set_parameter(x)

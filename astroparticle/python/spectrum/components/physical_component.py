"""Physical component.
"""
from astroparticle.python.spectrum.spectrum import Spectrum


class PhysicalComponent(Spectrum):

    def _set_parameter(self, x):
        raise NotImplementedError(
            "set_parameter not implemented")

    def set_parameter(self, x):
        self._set_parameter(x)

    def _parameter_size(self):
        raise NotImplementedError(
            "parameter_size not implemented")

    @property
    def parameter_size(self):
        return self._parameter_size

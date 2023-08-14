"""Sequence models.
"""
import tensorflow as tf

from astroparticle.python.spectrum.components.physical_component\
    import PhysicalComponent


class SequenceMultiplicative(PhysicalComponent):
    def __init__(self, components, dtype=tf.float32):
        self._components = components
        self.dtype = dtype

    def _forward(self, flux):
        flux = self._components[0](flux)
        for comp in self._components[1:]:
            flux = flux * comp(flux)
        return flux

    def _set_parameter(self, x):
        # x = tf.unstack(x, axis=-1)
        x = tf.convert_to_tensor(x, self.dtype)
        idx_start = 0
        for comp in self._components:
            idx_stop = idx_start + comp.parameter_size
            comp.set_parameter(x[..., idx_start:idx_stop])
